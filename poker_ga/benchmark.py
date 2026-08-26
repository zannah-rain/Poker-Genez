"""Cross-generation benchmark: plays the current population head-to-head
against a saved checkpoint from N generations ago, in 3-vs-3 tables, to give
a tangible, apples-to-apples measure of whether evolution is actually
improving play. The per-generation fitness number isn't useful for this --
it only ranks genomes against that generation's own random opponents, so a
"fitness" of 500 at generation 10 and 500 at generation 50 aren't
comparable. A direct match against a fixed past checkpoint is.

Poker has enough hand-to-hand variance that judging a match by a fixed
number of tables is unreliable: too few tables and a lucky/unlucky run of
cards decides the verdict as much as actual skill does; too many and
obvious results waste time that could go into more evolution. Instead,
`run_benchmark_until_resolved` runs a simple sequential test: it keeps
adding 3-vs-3 tables, tracking the current side's bb/100 edge as one
independent sample per table, until a confidence interval around the mean
edge no longer straddles zero (the sign of the edge is statistically
resolved) -- or until a hard cap on tables is reached, whichever comes
first.
"""

from __future__ import annotations

import math
from concurrent.futures import Executor, as_completed
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from cfr_tree import DEFAULT_MAX_STARTING_STACK_BB, DEFAULT_MIN_STARTING_STACK_BB
from game import GameConfig, SeatState, play_hand
from opponent_model import OpponentModel
from player import Player
from simulate import _executor_scope

SEATS_PER_SIDE = 3


def _draw_mirrored_stacks(
    rng: np.random.Generator, n: int, min_bb: float, max_bb: float, big_blind: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Draws `n` stack depths (in chips) independently and uniformly from
    [min_bb, max_bb] big blinds, then deals the identical multiset of
    depths to two sides, each independently shuffled -- so the two returned
    arrays always sum to the same total, however the depths themselves
    happen to be drawn. Used to seat both sides of a benchmark table so a
    lucky/unlucky spread of stack depths can never itself produce an edge,
    leaving play quality as the only thing that can."""
    stacks = rng.uniform(min_bb, max_bb, size=n) * big_blind
    return rng.permutation(stacks), rng.permutation(stacks)


def _play_side_match(
    current_pool: list[Player],
    checkpoint_pool: list[Player],
    game_config: GameConfig,
    rng: np.random.Generator,
    min_starting_stack_bb: float = DEFAULT_MIN_STARTING_STACK_BB,
    max_starting_stack_bb: float = DEFAULT_MAX_STARTING_STACK_BB,
) -> tuple[float, float, int, int]:
    """Plays one 3-vs-3 table to the hand cap. Any busted seat is refilled
    with a fresh player from its OWN side's pool (current or checkpoint), so
    the match stays a genuine 3v3 for the whole session instead of one side
    slowly being replaced by the other. Returns (current_net,
    checkpoint_net, current_hands, checkpoint_hands).

    Starting stacks are drawn from the same [min_starting_stack_bb,
    max_starting_stack_bb] spread that cfr_tree.traverse_hand trains on
    (defaulting to cfr_tree.DEFAULT_MIN/MAX_STARTING_STACK_BB), rather than
    a single fixed depth -- otherwise this benchmark would only ever judge
    play at one stack depth while training optimizes across the whole
    range, and a real edge or regression confined to shove/fold or
    deep-stack play would never show up here. The SEATS_PER_SIDE depths are
    drawn once and dealt as the identical multiset to both sides
    (independently shuffled per side), so the two sides always start a
    table with exactly the same total chips -- any edge that emerges is
    from play quality, never from one side getting luckier stack draws."""
    current_idx = rng.choice(len(current_pool), size=SEATS_PER_SIDE, replace=False)
    checkpoint_idx = rng.choice(len(checkpoint_pool), size=SEATS_PER_SIDE, replace=False)

    current_stacks, checkpoint_stacks = _draw_mirrored_stacks(
        rng, SEATS_PER_SIDE, min_starting_stack_bb, max_starting_stack_bb, game_config.big_blind,
    )

    seats = [SeatState(player=current_pool[i], stack=float(cs)) for i, cs in zip(current_idx, current_stacks)] + [
        SeatState(player=checkpoint_pool[i], stack=float(cs)) for i, cs in zip(checkpoint_idx, checkpoint_stacks)
    ]
    # Each seat's own cost basis -- what it was dealt at the start of its
    # current lifetime -- since that's no longer a single constant once
    # stacks are randomized per seat (and re-randomized on each bust below).
    basis = list(current_stacks) + list(checkpoint_stacks)
    sides = ["current"] * SEATS_PER_SIDE + ["checkpoint"] * SEATS_PER_SIDE
    order = rng.permutation(len(seats))  # so "current" isn't always dealt the button first
    seats = [seats[i] for i in order]
    sides = [sides[i] for i in order]
    basis = [float(basis[i]) for i in order]
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
                    current_net -= basis[i]
                    replacement = current_pool[int(rng.integers(0, len(current_pool)))]
                else:
                    checkpoint_net -= basis[i]
                    replacement = checkpoint_pool[int(rng.integers(0, len(checkpoint_pool)))]
                new_stack = float(rng.uniform(min_starting_stack_bb, max_starting_stack_bb) * game_config.big_blind)
                seats[i] = SeatState(player=replacement, stack=new_stack)
                basis[i] = new_stack

    for i, s in enumerate(seats):
        if sides[i] == "current":
            current_net += s.stack - basis[i]
        else:
            checkpoint_net += s.stack - basis[i]

    return current_net, checkpoint_net, current_hands, checkpoint_hands


def _play_one_bb100_sample(
    current_players: list[Player],
    checkpoint_players: list[Player],
    game_config: GameConfig,
    rng: np.random.Generator,
    min_starting_stack_bb: float = DEFAULT_MIN_STARTING_STACK_BB,
    max_starting_stack_bb: float = DEFAULT_MAX_STARTING_STACK_BB,
) -> float:
    """Plays one 3-vs-3 table and reduces it to the current side's bb/100 --
    the one independent sample run_benchmark_until_resolved accumulates.
    Deliberately a plain module-level function (not a closure/lambda) so it
    can be pickled and sent to a worker process."""
    cur_net, _chk_net, cur_hands, _chk_hands = _play_side_match(
        current_players, checkpoint_players, game_config, rng,
        min_starting_stack_bb, max_starting_stack_bb,
    )
    return (cur_net / game_config.big_blind) / cur_hands * 100.0 if cur_hands else 0.0


def _inverse_normal_cdf(p: float) -> float:
    """Approximates the inverse standard normal CDF (the probit function)
    via Peter Acklam's rational approximation (accurate to ~1.15e-9), so
    confidence intervals can be computed with only numpy/math -- no scipy
    dependency needed for this one calculation."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be strictly between 0 and 1, got {p}")

    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    p_low, p_high = 0.02425, 1 - 0.02425

    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
    )


def confidence_interval(samples: list[float], p_value: float) -> tuple[float, float]:
    """Two-sided (1 - p_value) confidence interval around the sample mean.
    Uses a normal approximation rather than a Student's t interval -- with
    the tens-to-hundreds of tables this is used for, the two are close
    enough that the extra machinery (and a scipy dependency) isn't worth
    it."""
    n = len(samples)
    if n < 2:
        raise ValueError("Need at least 2 samples to compute a confidence interval.")
    mean = float(np.mean(samples))
    std = float(np.std(samples, ddof=1))
    se = std / math.sqrt(n)
    z = _inverse_normal_cdf(1.0 - p_value / 2.0)
    margin = z * se
    return mean - margin, mean + margin


@dataclass
class BenchmarkOutcome:
    """Result of a sequential benchmark match against a checkpoint.
    `mean_bb_per_100`/`ci_low`/`ci_high` describe the CURRENT side's edge
    over the checkpoint -- the checkpoint's own edge is exactly the
    negation, since each table is zero-sum and both sides always see the
    same number of hands (see `_play_side_match`), so there's no need to
    track a separate distribution for it."""

    tables_played: int
    mean_bb_per_100: float
    ci_low: float
    ci_high: float
    resolved: bool  # True if the CI excluded 0 before hitting max_tables
    improved: bool  # the resolved verdict; conservatively False if unresolved
    hit_table_cap: bool  # True if max_tables was reached without resolving


def run_benchmark_until_resolved(
    current_players: list[Player],
    checkpoint_players: list[Player],
    game_config: GameConfig,
    rng: np.random.Generator,
    min_tables: int = 100,
    max_tables: int = 5000,
    table_batch: int = 50,
    p_value: float = 0.05,
    show_progress: bool = True,
    num_workers: int = 1,
    executor: Executor | None = None,
    min_starting_stack_bb: float = DEFAULT_MIN_STARTING_STACK_BB,
    max_starting_stack_bb: float = DEFAULT_MAX_STARTING_STACK_BB,
) -> BenchmarkOutcome:
    """Plays 3-vs-3 tables against the checkpoint one at a time, treating
    each table's current-side bb/100 as one independent sample. After the
    first `min_tables`, and again every `table_batch` tables after that, it
    checks whether the `1 - p_value` confidence interval around the mean
    sample still straddles zero: if not, the sign of the edge is considered
    statistically resolved and the match ends immediately (no need to keep
    playing once the direction is clear). If the CI still contains zero
    once `max_tables` is reached, the match ends anyway (`hit_table_cap`),
    conservatively reported as "not improved" -- the caller should treat
    that the same as a confirmed regression, since improvement was never
    actually established.

    Since this can take an unpredictable (and potentially large) number of
    tables to resolve, `show_progress` (on by default) both draws a tqdm bar
    across each batch and prints an interim status line -- tables played so
    far, the running bb/100 edge and its CI, resolved or not -- after every
    batch, so a slow run isn't silent while it works out which way the
    edge points.

    `num_workers`/`executor` play each batch's tables across that many
    worker processes instead of one at a time -- see
    simulate.run_generation's docstring for the same option there, including
    why a parallel run doesn't reproduce the exact same tables as
    num_workers=1 (the default) for the same seed.

    `min_starting_stack_bb`/`max_starting_stack_bb` set the range each
    table's per-seat starting stacks are drawn from -- see
    _play_side_match's docstring for why this should match the range
    training draws from (cfr_tree's own min/max-starting-stack-bb) rather
    than a single fixed depth."""
    if min_tables < 2:
        raise ValueError("min_tables must be at least 2 to compute a confidence interval.")
    if table_batch < 1:
        raise ValueError("table_batch must be at least 1.")

    samples: list[float] = []
    target = min_tables
    while True:
        batch_size = target - len(samples)
        with tqdm(
            total=batch_size, desc=f"benchmark ({len(samples)}-{target} tables)",
            unit="table", disable=not show_progress, leave=False,
        ) as progress:
            if num_workers <= 1:
                while len(samples) < target:
                    samples.append(
                        _play_one_bb100_sample(
                            current_players, checkpoint_players, game_config, rng,
                            min_starting_stack_bb, max_starting_stack_bb,
                        )
                    )
                    progress.update(1)
            else:
                batch_rngs = rng.spawn(batch_size)
                with _executor_scope(executor, num_workers) as pool:
                    futures = [
                        pool.submit(
                            _play_one_bb100_sample, current_players, checkpoint_players, game_config, table_rng,
                            min_starting_stack_bb, max_starting_stack_bb,
                        )
                        for table_rng in batch_rngs
                    ]
                    for future in as_completed(futures):
                        samples.append(future.result())
                        progress.update(1)

        ci_low, ci_high = confidence_interval(samples, p_value)
        mean = float(np.mean(samples))

        if ci_low > 0.0 or ci_high < 0.0:
            if show_progress:
                verdict = "IMPROVED" if ci_low > 0.0 else "REGRESSED"
                print(
                    f"    benchmark resolved after {len(samples)} tables | "
                    f"current edge {mean:+.2f} bb/100 (CI [{ci_low:+.2f}, {ci_high:+.2f}]) | {verdict}"
                )
            return BenchmarkOutcome(
                tables_played=len(samples), mean_bb_per_100=mean,
                ci_low=ci_low, ci_high=ci_high, resolved=True,
                improved=ci_low > 0.0, hit_table_cap=False,
            )
        if len(samples) >= max_tables:
            if show_progress:
                print(
                    f"    benchmark hit the {max_tables}-table cap without resolving | "
                    f"current edge {mean:+.2f} bb/100 (CI [{ci_low:+.2f}, {ci_high:+.2f}]) | "
                    "treating as NOT IMPROVED"
                )
            return BenchmarkOutcome(
                tables_played=len(samples), mean_bb_per_100=mean,
                ci_low=ci_low, ci_high=ci_high, resolved=False,
                improved=False, hit_table_cap=True,
            )
        if show_progress:
            print(
                f"    benchmark inconclusive after {len(samples)} tables | "
                f"current edge {mean:+.2f} bb/100 (CI [{ci_low:+.2f}, {ci_high:+.2f}]) | "
                f"still includes 0 -- playing {table_batch} more..."
            )
        target = min(len(samples) + table_batch, max_tables)
