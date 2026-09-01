"""CLI: train a Single Deep CFR advantage network over 6-max NLHE.

Every --benchmark-interval iterations, plays the current net head-to-head
against a pool of past snapshots (cfr_networks.clone) in a
statistically-resolved 3-vs-3 match (benchmark.py) -- a direct "did the
last --benchmark-interval iterations actually help" check. The pool is
built to look as much like the *training* data itself as this can manage,
on both axes that matter:

  - Membership: a pool member is kept for as long as the trainer's own
    reservoir still holds rows collected while it was the active net --
    tracked via each member's own start_iteration and the reservoir's own
    per-row `iteration` field (see cfr_reservoir.py's own docstring for why
    that's a *separate* field from `weights`/t, which cfr_tree.py's own
    path_weight can shrink well below a row's true iteration). Once a
    member's own attributable share of the reservoir's current total
    training weight fades below --benchmark-pool-min-weight-fraction, it's
    retired -- there's no fixed pool-size cap, because there's no reason to
    keep (or discard) a snapshot on a schedule when the reservoir itself
    already says how much it still actually matters to what's being
    trained right now.
  - Selection: each benchmark table draws its opponent from ONE pool
    member (never mixed within a table -- see
    benchmark.run_benchmark_until_resolved's own docstring), with
    probability proportional to that same attributable weight -- so a
    snapshot whose own era still dominates the reservoir gets tested
    against proportionally more, and one that's nearly aged out barely
    gets tested at all, mirroring how much the *net itself* has actually
    been shaped by each era rather than testing every surviving era
    equally.

This all exists because training is genuinely self-play (see cfr_tree.py's
own docstring): its convergence guarantee is about the *average* strategy
approaching equilibrium, not about monotonically beating any one earlier
version of itself head-to-head -- a run that's still improving can still
lose to a recent snapshot of itself purely by chance. A verdict pooled
across several past snapshots, weighted by how much each one still shapes
current training, is far less likely to flip on that kind of noise alone,
and stays honest about *which* past self is actually the relevant one to
beat. If a check doesn't show a resolved improvement, the update is undone
(Trainer.revert_to the pool's most recently confirmed-good member) and it
counts toward early stopping.

Example:
    python -m poker_ga.cfr_main --iterations 200 --traversals-per-iteration 200 --table-size 6
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import Executor, ProcessPoolExecutor
from contextlib import nullcontext

import numpy as np

import cfr_features
import cfr_networks
import cfr_policy
import cfr_reservoir
import cfr_tree
import gto
from benchmark import run_benchmark_until_resolved
from cfr_train import DeepCFRConfig, Trainer, run_iteration
from game import GameConfig
from player import Player

# benchmark.run_benchmark_until_resolved always plays 3-vs-3 tables
# (benchmark.SEATS_PER_SIDE), independent of --table-size.
BENCHMARK_SEATS_PER_SIDE = 3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Single Deep CFR poker strategy.")
    p.add_argument("--iterations", type=int, default=10_000)
    p.add_argument("--traversals-per-iteration", type=int, default=20)
    p.add_argument("--sgd-steps-per-iteration", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--dropout", type=float, default=cfr_networks.DEFAULT_DROPOUT,
        help="Advantage-net hidden-layer dropout rate (0 disables it). A no-op outside training "
        "-- predict()/SHAP both run in eval mode, where Dropout is the identity. Ignored (with a "
        "warning) if a checkpoint is reloaded -- see --hidden-sizes.",
    )
    p.add_argument(
        "--weight-decay", type=float, default=1e-5,
        help="Adam's L2 weight decay coefficient.",
    )
    p.add_argument(
        "--grad-clip-norm", type=float, default=5.0,
        help="Clips each _train_step's gradient to this global L2 norm before the optimizer step "
        "(0 disables clipping) -- guards against the loss-spike failure mode described in "
        "cfr_train.py's _train_step.",
    )
    p.add_argument("--reservoir-capacity", type=int, default=10_000_000)
    p.add_argument(
        "--num-equity-rollouts", type=int, default=cfr_tree.DEFAULT_NUM_EQUITY_ROLLOUTS,
        help="Terminal showdowns reached before the river (an early all-in) average this many "
        "possible board completions instead of dealing one random runout, as a lower-variance "
        "regression target -- exact whenever that many completions covers every possibility "
        "(e.g. a river-only completion), a Monte Carlo estimate otherwise (e.g. a preflop all-in).",
    )
    p.add_argument(
        "--min-starting-stack-bb", type=float, default=cfr_tree.DEFAULT_MIN_STARTING_STACK_BB,
        help="Every traversed hand independently redraws each seat's own starting stack, "
        "uniformly between this and --max-starting-stack-bb big blinds, rather than dealing every "
        "training hand from the same fixed depth -- a real tournament session has players sitting "
        "at a whole spread of stack depths at once, and a net that's only ever seen one fixed "
        "depth has no basis for the shove/fold and deep-stack decisions that depend on the others.",
    )
    p.add_argument(
        "--max-starting-stack-bb", type=float, default=cfr_tree.DEFAULT_MAX_STARTING_STACK_BB,
        help="See --min-starting-stack-bb.",
    )
    p.add_argument(
        "--hidden-sizes", type=str, default="128,128",
        help="Comma-separated advantage-net hidden layer sizes. Ignored (with a warning) if a "
        "checkpoint is reloaded -- a saved net's architecture can't change after the fact.",
    )
    p.add_argument(
        "--table-size", type=int, default=6,
        help="Ignored (with a warning) if a checkpoint is reloaded -- see --hidden-sizes.",
    )
    p.add_argument(
        "--feature-keys", type=str, default=None,
        help="Comma-separated features.FEATURE_NAMES keys the net conditions on. Defaults to "
        "cfr_features.DEFAULT_FEATURE_KEYS (every features.py key, excluding opponent-tendency "
        "reads -- see cfr_features.py). Ignored (with a warning) if a checkpoint is reloaded -- "
        "see --hidden-sizes.",
    )
    p.add_argument("--max-raises-per-street", type=int, default=4)
    p.add_argument("--min-raise-fraction-of-pot", type=float, default=0.25)
    p.add_argument(
        "--starting-stack", type=float, default=200.0,
        help="Only affects real (non-CFR) session play, e.g. the --benchmark-interval check -- "
        "training traversals ignore this and randomize their own starting stack instead, see "
        "--min-starting-stack-bb.",
    )
    p.add_argument("--small-blind", type=float, default=1.0)
    p.add_argument("--big-blind", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out-dir", type=str, default="cfr_runs")
    p.add_argument("--checkpoint-interval", type=int, default=1)
    p.add_argument(
        "--reload-previous", action=argparse.BooleanOptionalAction, default=True,
        help="Resume from <out-dir>/checkpoint_latest if it exists (pass --no-reload-previous to "
        "always start from a fresh random net).",
    )
    p.add_argument(
        "--benchmark-interval", type=int, default=0,
        help="The progress check: every this many iterations, play the current net "
        "head-to-head (3-vs-3 tables, benchmark.py) against a pool of past snapshots (see "
        "--benchmark-pool-min-weight-fraction), starting with just the net as it stood when this "
        "run started (iteration 0 for a fresh run, or wherever a reloaded checkpoint's own "
        "iteration count left off -- see --reload-previous) -- until the result is statistically "
        "resolved (see --benchmark-min/max-tables, --benchmark-p-value). A direct, apples-to-apples "
        "'did training actually help' check, unlike the training loss (see cfr_train.py's "
        "_train_step for why that's not comparable run over run). Set to 0 to disable.",
    )
    p.add_argument(
        "--benchmark-pool-min-weight-fraction", type=float, default=0.01,
        help="How the benchmark pool of past confirmed-good snapshots is maintained (see "
        "--benchmark-interval): each member's own share of the trainer's own reservoir's current "
        "total training weight is tracked (see cfr_reservoir.py's own `iterations` field and this "
        "module's own docstring) -- once a member's own share fades below this fraction, it's "
        "retired. No fixed pool-size cap -- a snapshot stays as long as the reservoir itself still "
        "holds a meaningful amount of the data it produced, however long or short that turns out to "
        "be, and is dropped once that's no longer true. The most recent member is always kept "
        "regardless of its own share (both the revert target for the next non-improving check and "
        "too newly promoted to have accumulated much reservoir weight yet). Each benchmark table "
        "then draws its opponent from ONE pool member (never mixed within a table -- see "
        "benchmark.run_benchmark_until_resolved), with probability proportional to that same share "
        "-- so testing mirrors, as closely as this can manage, how much each past era of training "
        "still actually shapes the current net, not just which eras happen to have survived at all. "
        "Falls back to a uniform draw across the pool whenever every member's own share is 0 (e.g. "
        "right after a fresh start, or immediately after reloading a reservoir saved before "
        "`iterations` existed -- see cfr_reservoir.UNKNOWN_ITERATION).",
    )
    p.add_argument(
        "--benchmark-min-tables", type=int, default=500,
        help="Minimum number of 3-vs-3 tables played against the checkpoint pool before checking "
        "whether the result is statistically resolved.",
    )
    p.add_argument(
        "--benchmark-max-tables", type=int, default=10_000,
        help="Hard cap on 3-vs-3 tables played in one benchmark check. If the confidence interval "
        "still straddles 0 at this point, the check ends anyway and is conservatively treated as "
        "'not improved'.",
    )
    p.add_argument(
        "--benchmark-table-batch", type=int, default=500,
        help="Additional tables played per round once --benchmark-min-tables isn't enough to "
        "resolve the confidence interval (repeats until it resolves or --benchmark-max-tables is hit).",
    )
    p.add_argument(
        "--benchmark-p-value", type=float, default=0.1,
        help="Improvement p-value threshold: the benchmark keeps playing tables until the "
        "(1 - N) confidence interval of the current net's bb/100 edge over the checkpoint no "
        "longer includes 0. Lower values demand stronger evidence before calling an iteration "
        "improved or regressed.",
    )
    p.add_argument(
        "--early-stop-patience", type=int, default=3,
        help="Whenever a --benchmark-interval check doesn't show a resolved improvement, the net's "
        "weights (and optimizer state) revert to that checkpoint. If this happens this many times "
        "in a row, training stops early rather than running out --iterations. 0 disables early "
        "stopping (still reverts on non-improvement, just keeps retrying indefinitely). Only "
        "applies when --benchmark-interval > 0.",
    )
    p.add_argument(
        "--gto-spots", action=argparse.BooleanOptionalAction, default=False,
        help="Play gto.GTO_SPOTS' fixed charts verbatim (for every seat, not just the traverser) "
        "instead of learning those specific decisions -- see gto.py's module docstring. The net "
        "still learns the optimal response everywhere else, including its own best reply to those "
        "fixed actions. Off by default: every decision is fully learned.",
    )
    p.add_argument(
        "--workers", type=int, default=0,
        help="Worker processes for the --benchmark-interval check specifically (CFR traversal "
        "itself is sequential in this version, so this only speeds up benchmarking). 1 is fully "
        "sequential (the default). 0 or negative means 'use every available CPU core'.",
    )
    return p.parse_args()


def _benchmark_players(
    net: cfr_networks.AdvantageNet, feature_keys: tuple[str, ...], label: str, id_offset: int,
    gto_spots: tuple[gto.GTOSpot, ...] = (),
) -> list[Player]:
    """A pool of BENCHMARK_SEATS_PER_SIDE Players all wrapping the same net
    -- run_benchmark_until_resolved draws its 3-a-side tables from a pool
    (refilling busted seats from it), so a single shared net just needs to
    appear in it that many times over."""
    return [
        Player(player_id=id_offset + i, genome=cfr_policy.DeepCFRPolicy(net, feature_keys, gto_spots=gto_spots), label=label)
        for i in range(BENCHMARK_SEATS_PER_SIDE)
    ]


# One (start_iteration, net) pair per pool member, oldest first --
# start_iteration is the outer training iteration at which that member was
# promoted (cloned from trainer.net right after a confirmed-improved
# benchmark check), so it also marks the end of the *previous* member's own
# span and the start of this one's -- see _benchmark_pool_weights.
BenchmarkPool = list[tuple[int, cfr_networks.AdvantageNet]]


def _benchmark_pool_member_path(checkpoint_path: str, i: int) -> str:
    return f"{checkpoint_path}_benchmark_pool_{i}"


def _benchmark_pool_starts_path(checkpoint_path: str) -> str:
    return f"{checkpoint_path}_benchmark_pool_starts.json"


def _save_benchmark_pool(
    benchmark_pool: BenchmarkPool, net_config: cfr_networks.AdvantageNetConfig, checkpoint_path: str,
) -> None:
    """Writes each benchmark_pool member's own weights (oldest first) to
    <checkpoint_path>_benchmark_pool_<i>.pt/.json, plus their own
    start_iterations together as one small JSON list
    (<checkpoint_path>_benchmark_pool_starts.json, index-aligned with the
    numbered members), alongside every trainer.save(checkpoint_path) call,
    so a resumed run's pool picks up exactly where this one left off
    instead of silently losing its own start_iteration attribution -- a
    resumed run must not differ from one that simply kept running in the
    same process (see _load_benchmark_pool). Removes any stale member left
    over from a previous, larger pool (e.g. more members have since been
    retired -- see _retire_stale_pool_members)."""
    for i, (_start, net) in enumerate(benchmark_pool):
        cfr_networks.save(net, net_config, _benchmark_pool_member_path(checkpoint_path, i))
    with open(_benchmark_pool_starts_path(checkpoint_path), "w", encoding="utf-8") as f:
        json.dump([start for start, _net in benchmark_pool], f)
    i = len(benchmark_pool)
    while os.path.exists(f"{_benchmark_pool_member_path(checkpoint_path, i)}.pt"):
        os.remove(f"{_benchmark_pool_member_path(checkpoint_path, i)}.pt")
        os.remove(f"{_benchmark_pool_member_path(checkpoint_path, i)}.json")
        i += 1


def _load_benchmark_pool(checkpoint_path: str) -> BenchmarkPool:
    """Reloads whatever benchmark pool _save_benchmark_pool last wrote for
    this checkpoint (oldest first), or [] if none exists -- a fresh run,
    --no-reload-previous, or a checkpoint saved before this existed all
    fall back to [] the same way (see _run_training, which then seeds a
    fresh single-entry pool exactly as it would for a brand new run). A
    pool saved before start_iteration tracking existed (nets on disk, but
    no `_benchmark_pool_starts.json`) falls back to treating every member
    as if it started at iteration 0 -- attribution against the reservoir
    will be wrong until the pool naturally reshapes itself around
    freshly-tracked rows (see cfr_reservoir.UNKNOWN_ITERATION for the same
    graceful-degradation spirit applied to individual reservoir rows)."""
    pool_nets = []
    i = 0
    while os.path.exists(f"{_benchmark_pool_member_path(checkpoint_path, i)}.pt"):
        net, _config = cfr_networks.load(_benchmark_pool_member_path(checkpoint_path, i))
        pool_nets.append(net)
        i += 1
    if not pool_nets:
        return []
    starts_path = _benchmark_pool_starts_path(checkpoint_path)
    if os.path.exists(starts_path):
        with open(starts_path, encoding="utf-8") as f:
            starts = json.load(f)
    else:
        starts = [0] * len(pool_nets)
    return list(zip(starts, pool_nets))


def _benchmark_pool_weights(
    benchmark_pool: BenchmarkPool, reservoir: cfr_reservoir.ReservoirBuffer, current_iteration: int,
) -> np.ndarray:
    """One weight per benchmark_pool member (same order), proportional to
    how much of the reservoir's own currently-held rows can be attributed
    to the span of iterations that member was the active net for -- see
    this module's own docstring. Member i "owns" every currently-held row
    whose own raw collection iteration (reservoir.iterations -- *not*
    reservoir.weights/t, which cfr_tree.py's own path_weight can shrink
    well below a row's true iteration -- see cfr_reservoir.py's own
    docstring) falls in [start_i, start_{i+1}) (or [start_last, +inf) for
    the most recent member); each such row's own contribution is exactly
    the Linear-CFR loss weight cfr_train._train_step would give it right
    now (row's own raw t / current_iteration, matching _train_step's own
    normalization exactly), summed. A row whose own iteration is
    cfr_reservoir.UNKNOWN_ITERATION (reservoir saved before that field
    existed) contributes to no member -- there's nothing honest to
    attribute it to -- so right after upgrading an existing run, every
    weight reads low (possibly all 0) until enough freshly-tracked rows
    accumulate to dominate; callers should treat an all-zero result as "not
    enough information yet" (see _resolve_benchmark_pool_weights), not
    "every member is equally stale"."""
    out = np.zeros(len(benchmark_pool))
    if not benchmark_pool or reservoir.size == 0:
        return out
    iterations = reservoir.iterations[: reservoir.size]
    row_weight = reservoir.weights[: reservoir.size] / max(current_iteration, 1)
    known = iterations != cfr_reservoir.UNKNOWN_ITERATION
    starts = [start for start, _net in benchmark_pool]
    for i, start in enumerate(starts):
        hi = starts[i + 1] if i + 1 < len(starts) else np.inf
        mask = known & (iterations >= start) & (iterations < hi)
        out[i] = float(row_weight[mask].sum())
    return out


def _retire_stale_pool_members(
    benchmark_pool: BenchmarkPool, pool_weights: np.ndarray, min_weight_fraction: float,
) -> BenchmarkPool:
    """Drops every benchmark_pool member -- except the most recent, always
    kept regardless of its own current weight (both because it's the
    revert target for the very next non-improving check, and because a
    just-promoted anchor may genuinely not have accumulated much reservoir
    weight yet even if it's about to) -- whose own current attributable
    weight share (`pool_weights`, see _benchmark_pool_weights) has faded
    below `min_weight_fraction` of the pool's own total: "close to 0
    attributable training weight" replaces a fixed pool-size cap as the
    retirement criterion (see this module's own docstring). A no-op (keeps
    everything) if the pool's own total weight is 0 -- nothing to compare
    shares against yet (e.g. right after a fresh start, or a resume with an
    all-unknown-provenance reloaded reservoir -- see
    _benchmark_pool_weights)."""
    if len(benchmark_pool) <= 1:
        return benchmark_pool
    total = float(pool_weights.sum())
    if total <= 0.0:
        return benchmark_pool
    shares = pool_weights / total
    last = len(benchmark_pool) - 1
    return [member for i, member in enumerate(benchmark_pool) if i == last or shares[i] >= min_weight_fraction]


def _resolve_benchmark_pool(
    benchmark_pool: BenchmarkPool, reservoir: cfr_reservoir.ReservoirBuffer, current_iteration: int,
    min_weight_fraction: float,
) -> tuple[BenchmarkPool, np.ndarray]:
    """Retires every stale member from `benchmark_pool` (see
    _retire_stale_pool_members) based on its own current attributable
    weight (see _benchmark_pool_weights), then returns (retained_pool,
    normalized_sampling_weights) for whatever's left -- normalized_
    sampling_weights sums to 1, falling back to a uniform draw across the
    retained pool if every member's own attributable weight is still 0
    (nothing to distinguish them by yet -- see _benchmark_pool_weights's
    own docstring for when that happens)."""
    weights = _benchmark_pool_weights(benchmark_pool, reservoir, current_iteration)
    retained = _retire_stale_pool_members(benchmark_pool, weights, min_weight_fraction)
    if len(retained) != len(benchmark_pool):
        weights = _benchmark_pool_weights(retained, reservoir, current_iteration)
    total = float(weights.sum())
    normalized = weights / total if total > 0.0 else np.full(len(retained), 1.0 / len(retained))
    return retained, normalized


def _reload_checkpoint(
    checkpoint_path: str, reload_previous: bool, config: DeepCFRConfig, rng: np.random.Generator,
) -> tuple[
    cfr_networks.AdvantageNet | None, cfr_reservoir.ReservoirBuffer | None, dict | None, int, DeepCFRConfig,
]:
    """Returns (net_to_resume_from_or_None, reservoir_to_resume_from_or_None,
    optimizer_state_to_resume_from_or_None, completed_iterations, config),
    where `config` has been updated in place to match the reloaded net's
    own architecture if one was found -- a saved net's input/hidden layer
    shapes can't change after the fact, so the checkpoint's own
    feature_keys/hidden_sizes/table_size always win over whatever
    --feature-keys/--hidden-sizes/--table-size were passed on the command
    line. The reservoir (Trainer.save's `<path>.npz`) and optimizer
    state/completed_iterations (Trainer.save's `<path>_trainer_state.pt`)
    are reloaded alongside the net whenever they exist -- see Trainer.save's
    docstring for why all three are always written together -- but a net
    checkpoint from before one of them was added won't have it, so each is
    allowed to come back None/0 (a fresh empty reservoir, a fresh optimizer,
    t=0) without treating it as an error. completed_iterations is always
    part of the training loop's iteration numbering (0 whenever no trainer
    state was found), never left dangling, since a mismatch between it and
    the reloaded reservoir's own stored `t` values is exactly the bug that
    caused training loss to spike after a resume -- see Trainer's
    docstring."""
    if not reload_previous:
        print("--no-reload-previous passed; starting from a fresh random net and reservoir.")
        return None, None, None, 0, config
    if not (os.path.exists(f"{checkpoint_path}.pt") and os.path.exists(f"{checkpoint_path}.json")):
        print(f"No previous checkpoint found at {checkpoint_path}; starting from scratch.")
        return None, None, None, 0, config

    try:
        net, loaded_config = cfr_networks.load(checkpoint_path)
    except Exception as exc:
        print(f"Warning: could not reload checkpoint from {checkpoint_path} ({exc}); starting from scratch.")
        return None, None, None, 0, config

    print(
        f"Reloaded checkpoint from {checkpoint_path} "
        f"({len(loaded_config.feature_keys)} features, hidden {loaded_config.hidden_sizes}, "
        f"table_size {loaded_config.table_size})."
    )
    if (
        loaded_config.feature_keys != config.feature_keys
        or loaded_config.hidden_sizes != config.hidden_sizes
        or loaded_config.table_size != config.table_size
        or loaded_config.dropout != config.dropout
    ):
        print(
            "Warning: reloaded checkpoint's architecture differs from the requested "
            "--feature-keys/--hidden-sizes/--table-size/--dropout -- using the checkpoint's own "
            "architecture instead."
        )
    config.feature_keys = loaded_config.feature_keys
    config.hidden_sizes = loaded_config.hidden_sizes
    config.table_size = loaded_config.table_size
    config.dropout = loaded_config.dropout

    reservoir = None
    if os.path.exists(f"{checkpoint_path}.npz"):
        try:
            reservoir = cfr_reservoir.ReservoirBuffer.load(checkpoint_path, rng)
        except Exception as exc:
            print(f"Warning: could not reload reservoir from {checkpoint_path} ({exc}); starting with an empty one.")
        else:
            print(
                f"Reloaded reservoir from {checkpoint_path} "
                f"({reservoir.size}/{reservoir.capacity} samples, {reservoir.n_seen} seen total)."
            )
            if config.reservoir_capacity > reservoir.capacity:
                print(
                    f"Requested --reservoir-capacity ({config.reservoir_capacity}) exceeds the reloaded "
                    f"reservoir's own capacity ({reservoir.capacity}) -- growing it to match rather than "
                    "leaving the extra requested capacity unused."
                )
                reservoir.grow(config.reservoir_capacity)
            elif reservoir.capacity != config.reservoir_capacity:
                print(
                    f"Warning: reloaded reservoir's capacity ({reservoir.capacity}) differs from the "
                    f"requested --reservoir-capacity ({config.reservoir_capacity}) -- using the "
                    "reservoir's own (larger) capacity instead."
                )
                config.reservoir_capacity = reservoir.capacity
    else:
        print(f"No reservoir checkpoint found at {checkpoint_path}.npz; starting with an empty reservoir.")

    optimizer_state = None
    completed_iterations = 0
    try:
        optimizer_state, completed_iterations = Trainer.load_trainer_state(checkpoint_path)
    except Exception as exc:
        print(
            f"Warning: could not reload optimizer/iteration state from {checkpoint_path} ({exc}); "
            "resuming with a fresh optimizer at iteration 0 (this will re-spike the reported loss "
            "the same way a missing trainer-state file does -- see Trainer's docstring)."
        )
    else:
        if optimizer_state is not None:
            print(f"Reloaded optimizer state and iteration count ({completed_iterations}) from {checkpoint_path}.")
        else:
            print(
                f"No optimizer/iteration state found at {checkpoint_path} (checkpoint predates it); "
                "resuming with a fresh optimizer at iteration 0."
            )

    return net, reservoir, optimizer_state, completed_iterations, config


def _run_training(args: argparse.Namespace, rng: np.random.Generator, num_workers: int, executor: Executor | None) -> None:
    feature_keys = (
        tuple(k.strip() for k in args.feature_keys.split(",")) if args.feature_keys
        else cfr_features.DEFAULT_FEATURE_KEYS
    )
    hidden_sizes = tuple(int(x.strip()) for x in args.hidden_sizes.split(","))

    game_config = GameConfig(
        small_blind=args.small_blind,
        big_blind=args.big_blind,
        starting_stack=args.starting_stack,
        max_raises_per_street=args.max_raises_per_street,
        min_raise_fraction_of_pot=args.min_raise_fraction_of_pot,
    )
    config = DeepCFRConfig(
        feature_keys=feature_keys,
        hidden_sizes=hidden_sizes,
        table_size=args.table_size,
        iterations=args.iterations,
        traversals_per_iteration=args.traversals_per_iteration,
        sgd_steps_per_iteration=args.sgd_steps_per_iteration,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        reservoir_capacity=args.reservoir_capacity,
        num_equity_rollouts=args.num_equity_rollouts,
        min_starting_stack_bb=args.min_starting_stack_bb,
        max_starting_stack_bb=args.max_starting_stack_bb,
        game_config=game_config,
        gto_spots=gto.GTO_SPOTS if args.gto_spots else (),
    )

    os.makedirs(args.out_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.out_dir, "checkpoint_latest")
    initial_net, initial_reservoir, initial_optimizer_state, completed_iterations, config = _reload_checkpoint(
        checkpoint_path, args.reload_previous, config, rng,
    )
    trainer = Trainer.new(
        config, rng, initial_net=initial_net, initial_reservoir=initial_reservoir,
        initial_optimizer_state=initial_optimizer_state, completed_iterations=completed_iterations,
    )

    # The --benchmark-interval pool: reloaded from wherever a previous run
    # of this same checkpoint last left it (_load_benchmark_pool -- gated on
    # --reload-previous the same way _reload_checkpoint's own net/reservoir/
    # optimizer-state reload is, so --no-reload-previous starts completely
    # fresh across the board), so a resumed run's pool picks up exactly
    # where it left off rather than silently collapsing back to a single
    # entry -- a resumed run must not differ from one that simply kept
    # running in the same process. Falls back to a single-entry pool
    # holding whatever net this run's iteration loop below is about to start
    # from (freshly random, or reloaded -- including trainer.completed_iterations
    # iterations of prior training, since that's carried forward too -- see
    # _reload_checkpoint) whenever nothing was reloaded (a fresh run,
    # --no-reload-previous, or a checkpoint saved before this pool existed).
    # Gains one more entry each time a check shows improvement, and loses
    # whichever ones have faded below --benchmark-pool-min-weight-fraction
    # of the reservoir's own current attributable weight (see this module's
    # own docstring and _resolve_benchmark_pool) -- no fixed size cap.
    benchmark_pool: BenchmarkPool = []
    if args.benchmark_interval > 0:
        if args.reload_previous:
            benchmark_pool = _load_benchmark_pool(checkpoint_path)
            if benchmark_pool:
                print(f"Reloaded a {len(benchmark_pool)}-entry benchmark pool from {checkpoint_path}.")
        if not benchmark_pool:
            benchmark_pool = [(trainer.completed_iterations, cfr_networks.clone(trainer.net))]

    consecutive_non_improvements = 0

    # --iterations is "how many more iterations to run", not a total --
    # continuing the count from trainer.completed_iterations (0 for a fresh
    # net) rather than restarting at 1 keeps it on the same scale as any
    # reloaded reservoir samples' own stored `t` values -- see Trainer's
    # docstring for why a resumed run must not renumber from 1.
    start_iteration = trainer.completed_iterations
    for iteration in range(start_iteration + 1, start_iteration + args.iterations + 1):
        t0 = time.time()
        mean_loss = run_iteration(trainer, rng, iteration)
        elapsed = time.time() - t0
        print(
            f"iter {iteration:4d} | reservoir {len(trainer.reservoir):7d}/{config.reservoir_capacity} | "
            f"mean loss {mean_loss:10.4f} | {elapsed:5.1f}s"
        )

        if args.checkpoint_interval > 0 and iteration % args.checkpoint_interval == 0:
            trainer.save(checkpoint_path)
            if benchmark_pool:
                _save_benchmark_pool(benchmark_pool, trainer.net_config(), checkpoint_path)

        if args.benchmark_interval > 0 and iteration % args.benchmark_interval == 0:
            benchmark_pool, pool_sampling_weights = _resolve_benchmark_pool(
                benchmark_pool, trainer.reservoir, iteration, args.benchmark_pool_min_weight_fraction,
            )
            current_players = _benchmark_players(trainer.net, config.feature_keys, "current", id_offset=0, gto_spots=config.gto_spots)
            checkpoint_pools = [
                _benchmark_players(
                    anchor, config.feature_keys, f"checkpoint_{i}",
                    id_offset=-100 * (i + 1), gto_spots=config.gto_spots,
                )
                for i, (_start, anchor) in enumerate(benchmark_pool)
            ]
            outcome = run_benchmark_until_resolved(
                current_players, checkpoint_pools, game_config, rng,
                min_tables=args.benchmark_min_tables, max_tables=args.benchmark_max_tables,
                table_batch=args.benchmark_table_batch, p_value=args.benchmark_p_value,
                show_progress=True, num_workers=num_workers, executor=executor,
                min_starting_stack_bb=config.min_starting_stack_bb,
                max_starting_stack_bb=config.max_starting_stack_bb,
                checkpoint_pool_weights=pool_sampling_weights,
            )
            if outcome.resolved:
                verdict = "IMPROVED" if outcome.improved else "REGRESSED"
            else:
                verdict = "INCONCLUSIVE (table cap hit; treated as not improved)"
            confidence_pct = (1.0 - args.benchmark_p_value) * 100.0
            weights_label = ", ".join(f"{w:.0%}" for w in pool_sampling_weights)
            print(
                f"         | benchmark vs pool of {len(benchmark_pool)} past snapshot(s) "
                f"(sampled {weights_label}) | {outcome.tables_played} tables | "
                f"current edge {outcome.mean_bb_per_100:+7.2f} bb/100 "
                f"({confidence_pct:.0f}% CI [{outcome.ci_low:+7.2f}, {outcome.ci_high:+7.2f}]) | {verdict}"
            )
            if outcome.improved:
                consecutive_non_improvements = 0
                benchmark_pool.append((iteration, cfr_networks.clone(trainer.net)))
            else:
                consecutive_non_improvements += 1
                patience_label = args.early_stop_patience if args.early_stop_patience > 0 else "unlimited"
                print(
                    f"         | this update didn't help -- reverting net to the pool's most recently "
                    f"confirmed-good snapshot ({consecutive_non_improvements}/{patience_label} "
                    "consecutive non-improvements)"
                )
                trainer.revert_to(benchmark_pool[-1][1])
                trainer.save(checkpoint_path)
                _save_benchmark_pool(benchmark_pool, trainer.net_config(), checkpoint_path)
                if args.early_stop_patience > 0 and consecutive_non_improvements >= args.early_stop_patience:
                    print(
                        f"Early stopping: no improvement for {args.early_stop_patience} "
                        "consecutive benchmark checks."
                    )
                    break

    trainer.save(checkpoint_path)
    if benchmark_pool:
        _save_benchmark_pool(benchmark_pool, trainer.net_config(), checkpoint_path)
    print(f"Done. Checkpoint saved to {checkpoint_path}.{{pt,json,npz}}")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    num_workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)

    pool_context = ProcessPoolExecutor(max_workers=num_workers) if num_workers > 1 else nullcontext(None)
    with pool_context as executor:
        _run_training(args, rng, num_workers, executor)


if __name__ == "__main__":
    main()
