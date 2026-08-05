"""CLI: train a Single Deep CFR advantage network over 6-max NLHE.

Every --benchmark-interval iterations, plays the current net head-to-head
against a snapshot (cfr_networks.clone) taken --benchmark-interval
iterations ago in a statistically-resolved 3-vs-3 match (benchmark.py) -- a
direct "did the last --benchmark-interval iterations actually help" check.
That snapshot is an anchor, not a rolling one iteration behind: it only
advances to the current net when a check shows improvement, so a run of
non-improving checks keeps comparing against the last net that was
actually beaten. If a check doesn't show a resolved improvement, the
update is undone (Trainer.revert_to the anchor snapshot) and it counts
toward early stopping.

Example:
    python -m poker_ga.cfr_main --iterations 200 --traversals-per-iteration 200 --table-size 6
"""

from __future__ import annotations

import argparse
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
    p.add_argument("--reservoir-capacity", type=int, default=1_000_000)
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
        "head-to-head (3-vs-3 tables, benchmark.py) against a snapshot taken this many iterations "
        "ago -- e.g. at 100, the first check is iteration 100 vs the net as it stood when this run "
        "started (iteration 0 for a fresh run, or wherever a reloaded checkpoint's own iteration "
        "count left off -- see --reload-previous) -- until the result is statistically resolved "
        "(see --benchmark-min/max-tables, "
        "--benchmark-p-value). That snapshot only advances to the current net when a check shows "
        "improvement, so it stays anchored on the last net actually beaten across any non-improving "
        "checks in between. A direct, apples-to-apples 'did training actually help' check, unlike "
        "the training loss (see cfr_train.py's _train_step for why that's not comparable run over "
        "run). Set to 0 to disable.",
    )
    p.add_argument(
        "--benchmark-min-tables", type=int, default=200,
        help="Minimum number of 3-vs-3 tables played against the checkpoint before checking "
        "whether the result is statistically resolved.",
    )
    p.add_argument(
        "--benchmark-max-tables", type=int, default=10_000,
        help="Hard cap on 3-vs-3 tables played in one benchmark check. If the confidence interval "
        "still straddles 0 at this point, the check ends anyway and is conservatively treated as "
        "'not improved'.",
    )
    p.add_argument(
        "--benchmark-table-batch", type=int, default=200,
        help="Additional tables played per round once --benchmark-min-tables isn't enough to "
        "resolve the confidence interval (repeats until it resolves or --benchmark-max-tables is hit).",
    )
    p.add_argument(
        "--benchmark-p-value", type=float, default=0.2,
        help="Improvement p-value threshold: the benchmark keeps playing tables until the "
        "(1 - N) confidence interval of the current net's bb/100 edge over the checkpoint no "
        "longer includes 0. Lower values demand stronger evidence before calling an iteration "
        "improved or regressed.",
    )
    p.add_argument(
        "--early-stop-patience", type=int, default=2,
        help="Whenever a --benchmark-interval check doesn't show a resolved improvement, the net's "
        "weights (and optimizer state) revert to that checkpoint. If this happens this many times "
        "in a row, training stops early rather than running out --iterations. 0 disables early "
        "stopping (still reverts on non-improvement, just keeps retrying indefinitely). Only "
        "applies when --benchmark-interval > 0.",
    )
    p.add_argument(
        "--workers", type=int, default=0,
        help="Worker processes for the --benchmark-interval check specifically (CFR traversal "
        "itself is sequential in this version, so this only speeds up benchmarking). 1 is fully "
        "sequential (the default). 0 or negative means 'use every available CPU core'.",
    )
    return p.parse_args()


def _benchmark_players(net: cfr_networks.AdvantageNet, feature_keys: tuple[str, ...], label: str, id_offset: int) -> list[Player]:
    """A pool of BENCHMARK_SEATS_PER_SIDE Players all wrapping the same net
    -- run_benchmark_until_resolved draws its 3-a-side tables from a pool
    (refilling busted seats from it), so a single shared net just needs to
    appear in it that many times over."""
    return [
        Player(player_id=id_offset + i, genome=cfr_policy.DeepCFRPolicy(net, feature_keys), label=label)
        for i in range(BENCHMARK_SEATS_PER_SIDE)
    ]


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
            if reservoir.capacity != config.reservoir_capacity:
                print(
                    f"Warning: reloaded reservoir's capacity ({reservoir.capacity}) differs from the "
                    f"requested --reservoir-capacity ({config.reservoir_capacity}) -- using the "
                    "reservoir's own capacity instead."
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

    # The --benchmark-interval anchor: starts at whatever net this run's
    # iteration loop below is about to start from (freshly random, or
    # reloaded -- including trainer.completed_iterations iterations of
    # prior training, since that's carried forward too -- see
    # _reload_checkpoint) and only advances to the current net when a check
    # shows improvement -- see --benchmark-interval's help text.
    benchmark_net = cfr_networks.clone(trainer.net) if args.benchmark_interval > 0 else None

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

        if args.benchmark_interval > 0 and iteration % args.benchmark_interval == 0:
            current_players = _benchmark_players(trainer.net, config.feature_keys, "current", id_offset=0)
            checkpoint_players = _benchmark_players(benchmark_net, config.feature_keys, "checkpoint", id_offset=-100)
            outcome = run_benchmark_until_resolved(
                current_players, checkpoint_players, game_config, rng,
                min_tables=args.benchmark_min_tables, max_tables=args.benchmark_max_tables,
                table_batch=args.benchmark_table_batch, p_value=args.benchmark_p_value,
                show_progress=True, num_workers=num_workers, executor=executor,
            )
            if outcome.resolved:
                verdict = "IMPROVED" if outcome.improved else "REGRESSED"
            else:
                verdict = "INCONCLUSIVE (table cap hit; treated as not improved)"
            confidence_pct = (1.0 - args.benchmark_p_value) * 100.0
            print(
                f"         | benchmark vs checkpoint ({args.benchmark_interval} iters ago) | "
                f"{outcome.tables_played} tables | current edge {outcome.mean_bb_per_100:+7.2f} bb/100 "
                f"({confidence_pct:.0f}% CI [{outcome.ci_low:+7.2f}, {outcome.ci_high:+7.2f}]) | {verdict}"
            )
            if outcome.improved:
                consecutive_non_improvements = 0
                benchmark_net = cfr_networks.clone(trainer.net)
            else:
                consecutive_non_improvements += 1
                patience_label = args.early_stop_patience if args.early_stop_patience > 0 else "unlimited"
                print(
                    f"         | this update didn't help -- reverting net to its benchmark checkpoint "
                    f"({consecutive_non_improvements}/{patience_label} consecutive non-improvements)"
                )
                trainer.revert_to(benchmark_net)
                trainer.save(checkpoint_path)
                if args.early_stop_patience > 0 and consecutive_non_improvements >= args.early_stop_patience:
                    print(
                        f"Early stopping: no improvement for {args.early_stop_patience} "
                        "consecutive benchmark checks."
                    )
                    break

    trainer.save(checkpoint_path)
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
