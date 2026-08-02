"""CLI: train a Single Deep CFR advantage network over 6-max NLHE.

Mirrors main.py's (the GA's) training-loop conveniences: resumes from the
previous run's checkpoint if one exists, and, every --benchmark-interval
iterations, snapshots the net's weights right before that iteration's
training update (cfr_networks.clone) and plays the post-update net
head-to-head against that pre-update snapshot in a statistically-resolved
3-vs-3 match (benchmark.py) -- a direct "did this update actually help"
check, primarily against the net's own immediately-preceding self. If the
benchmark doesn't show a resolved improvement, the update is undone
(Trainer.revert_to the pre-update snapshot) and it counts toward early
stopping, the same way main.py reverts to (and stops retrying past) a
non-improving GA checkpoint.

Optionally (--eval-interval), it can *also* be checked against real GA
genomes -- loaded from a saved population (--eval-genome-path, e.g. a
main.py run's latest_population.json) rather than freshly random ones --
as a secondary "is this any good against a real strategy, not just against
itself" sense check; see --eval-interval's help text.

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
from genome import Genome, load_population
from player import Player
from simulate import SimConfig
from tournament import run_final_tournament

# benchmark.run_benchmark_until_resolved always plays 3-vs-3 tables
# (benchmark.SEATS_PER_SIDE), independent of --table-size -- matching
# main.py, which benchmarks the GA the same fixed way regardless of its own
# --population/table configuration.
BENCHMARK_SEATS_PER_SIDE = 3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Single Deep CFR poker strategy.")
    p.add_argument("--iterations", type=int, default=10_000)
    p.add_argument("--traversals-per-iteration", type=int, default=20)
    p.add_argument("--sgd-steps-per-iteration", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--reservoir-capacity", type=int, default=1_000_000)
    p.add_argument(
        "--num-equity-rollouts", type=int, default=cfr_tree.DEFAULT_NUM_EQUITY_ROLLOUTS,
        help="Terminal showdowns reached before the river (an early all-in) average this many "
        "possible board completions instead of dealing one random runout, as a lower-variance "
        "regression target -- exact whenever that many completions covers every possibility "
        "(e.g. a river-only completion), a Monte Carlo estimate otherwise (e.g. a preflop all-in).",
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
    p.add_argument("--starting-stack", type=float, default=200.0)
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
        "--eval-interval", type=int, default=0,
        help="Optional, secondary check (see --benchmark-interval for the primary one): every "
        "this many iterations, benchmark the current net (bb/100) against a table of GA genomes "
        "via tournament.run_final_tournament -- an absolute 'is this any good against a real "
        "strategy' sense check, since --benchmark-interval only measures improvement relative to "
        "this same run's own past self. Opponents are loaded from --eval-genome-path if that file "
        "exists, else freshly random genomes (genome.Genome.random). 0 disables it.",
    )
    p.add_argument("--eval-rounds", type=int, default=20, help="Rounds of re-seating for each --eval-interval check.")
    p.add_argument(
        "--eval-genome-path", type=str, default=os.path.join("runs", "latest_population.json"),
        help="Population file (genome.save_population format) to draw --eval-interval's opponents "
        "from -- e.g. main.py's --out-dir's latest_population.json (the default here matches "
        "main.py's own default --out-dir, so a GA run's latest population is used automatically if "
        "one exists). Re-read fresh on every eval check, so a concurrently-running GA's progress is "
        "picked up. Falls back to freshly random genomes (with a warning) if the file doesn't exist.",
    )
    p.add_argument(
        "--benchmark-interval", type=int, default=100,
        help="The primary progress check: every this many iterations, snapshot the net's weights "
        "*before* that iteration's training update, run the update, then play the post-update net "
        "head-to-head against that pre-update snapshot (3-vs-3 tables, benchmark.py) until the "
        "result is statistically resolved (see --benchmark-min/max-tables, --benchmark-p-value) -- "
        "a direct, apples-to-apples 'did this update actually help' check, unlike the training loss "
        "(see cfr_train.py's _train_step for why that's not comparable run over run). Set to 0 to "
        "disable.",
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
        "itself is sequential in this version, unlike main.py's fully parallel GA evaluation, so "
        "this only speeds up benchmarking). 1 is fully sequential (the default). 0 or negative "
        "means 'use every available CPU core'.",
    )
    p.add_argument(
        "--feature-importance", action=argparse.BooleanOptionalAction, default=True,
        help="Every iteration, print the --feature-importance-top-n most- and least-important "
        "features by mean |SHAP value| (cfr_networks.mean_shap_contributions) over a random "
        "subsample of the reservoir -- pass --no-feature-importance to skip it (it adds a few "
        "seconds per iteration at the defaults below; see --feature-importance-sample-size).",
    )
    p.add_argument("--feature-importance-top-n", type=int, default=5, help="How many top/bottom features to print.")
    p.add_argument(
        "--feature-importance-sample-size", type=int, default=200,
        help="Reservoir samples explained per --feature-importance check. Cost scales roughly with "
        "this times --feature-importance-nsamples -- the defaults run in a couple of seconds "
        "against a (128, 128) net on CPU; pushing this into the thousands for a smoother estimate "
        "costs closer to a minute.",
    )
    p.add_argument(
        "--feature-importance-background-size", type=int, default=20,
        help="Background (reference) samples shap.GradientExplainer interpolates from -- see "
        "cfr_networks.mean_shap_contributions.",
    )
    p.add_argument(
        "--feature-importance-nsamples", type=int, default=20,
        help="Interpolation samples per explained point (shap.GradientExplainer's own `nsamples`) -- "
        "see --feature-importance-sample-size for the combined cost.",
    )
    return p.parse_args()


def _load_eval_opponents(genome_population_path: str, count: int, rng: np.random.Generator) -> tuple[list[Player], str]:
    """`count` opponent Players for an --eval-interval check, drawn from
    `genome_population_path` if it exists (re-read fresh every call, so a
    concurrently-running GA's progress gets picked up), else freshly random
    genomes. Returns (players, source_label) -- the label just distinguishes
    the two cases in the printed eval line."""
    if genome_population_path and os.path.exists(genome_population_path):
        try:
            genomes = load_population(genome_population_path, rng)
            if genomes:
                chosen_idx = rng.choice(len(genomes), size=count, replace=len(genomes) < count)
                players = [
                    Player(player_id=i + 1, genome=genomes[j], label=f"ga_{i + 1}")
                    for i, j in enumerate(chosen_idx)
                ]
                return players, f"GA genomes from {genome_population_path}"
        except Exception as exc:
            print(f"Warning: could not load genome population from {genome_population_path} ({exc}); using random GA genomes instead.")
    return (
        [Player(player_id=i + 1, genome=Genome.random(rng), label=f"random_ga_{i + 1}") for i in range(count)],
        "freshly random GA genomes",
    )


def _make_eval_fn(
    game_config: GameConfig, table_size: int, feature_keys: tuple[str, ...], eval_rounds: int,
    eval_genome_path: str, rng: np.random.Generator,
):
    def eval_fn(net, iteration: int) -> None:
        cfr_player = Player(player_id=0, genome=cfr_policy.DeepCFRPolicy(net, feature_keys), label="DeepCFR")
        opponents, source = _load_eval_opponents(eval_genome_path, table_size - 1, rng)
        sim_config = SimConfig(rounds_per_generation=eval_rounds, table_size=table_size)
        stats = run_final_tournament(
            [cfr_player, *opponents], game_config, sim_config, rng, show_progress=False,
        )
        bb_per_100 = stats[cfr_player.player_id].bb_per_100(game_config.big_blind)
        print(f"         | eval @ iter {iteration}: DeepCFR vs {source} = {bb_per_100:+.2f} bb/100")

    return eval_fn


def _print_feature_importance(
    trainer: Trainer, rng: np.random.Generator, top_n: int, sample_size: int, background_size: int, nsamples: int,
) -> None:
    contributions = cfr_networks.mean_shap_contributions(
        trainer.net, trainer.reservoir, trainer.config.feature_keys, rng,
        sample_size=sample_size, background_size=background_size, nsamples=nsamples,
    )
    if not contributions:
        return  # reservoir still empty (e.g. iteration 1 with a tiny traversals_per_iteration)
    top = contributions[:top_n]
    bottom = contributions[-top_n:] if len(contributions) > top_n else []
    top_str = ", ".join(f"{k}={v:.4f}" for k, v in top)
    print(f"         | feature importance (top {len(top)})    | {top_str}")
    if bottom:
        bottom_str = ", ".join(f"{k}={v:.4f}" for k, v in reversed(bottom))
        print(f"         | feature importance (bottom {len(bottom)}) | {bottom_str}")


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
) -> tuple[cfr_networks.AdvantageNet | None, cfr_reservoir.ReservoirBuffer | None, DeepCFRConfig]:
    """Returns (net_to_resume_from_or_None, reservoir_to_resume_from_or_None,
    config), where `config` has been updated in place to match the reloaded
    net's own architecture if one was found -- a saved net's input/hidden
    layer shapes can't change after the fact, so the checkpoint's own
    feature_keys/hidden_sizes/table_size always win over whatever
    --feature-keys/--hidden-sizes/--table-size were passed on the command
    line. The reservoir (Trainer.save's `<path>.npz`) is reloaded alongside
    the net whenever both exist -- see Trainer.save's docstring for why
    they're always written together -- but a net checkpoint from before this
    was added won't have one, so that half is allowed to come back None
    (a fresh empty reservoir) without treating it as an error."""
    if not reload_previous:
        print("--no-reload-previous passed; starting from a fresh random net and reservoir.")
        return None, None, config
    if not (os.path.exists(f"{checkpoint_path}.pt") and os.path.exists(f"{checkpoint_path}.json")):
        print(f"No previous checkpoint found at {checkpoint_path}; starting from scratch.")
        return None, None, config

    try:
        net, loaded_config = cfr_networks.load(checkpoint_path)
    except Exception as exc:
        print(f"Warning: could not reload checkpoint from {checkpoint_path} ({exc}); starting from scratch.")
        return None, None, config

    print(
        f"Reloaded checkpoint from {checkpoint_path} "
        f"({len(loaded_config.feature_keys)} features, hidden {loaded_config.hidden_sizes}, "
        f"table_size {loaded_config.table_size})."
    )
    if (
        loaded_config.feature_keys != config.feature_keys
        or loaded_config.hidden_sizes != config.hidden_sizes
        or loaded_config.table_size != config.table_size
    ):
        print(
            "Warning: reloaded checkpoint's architecture differs from the requested "
            "--feature-keys/--hidden-sizes/--table-size -- using the checkpoint's own "
            "architecture instead."
        )
    config.feature_keys = loaded_config.feature_keys
    config.hidden_sizes = loaded_config.hidden_sizes
    config.table_size = loaded_config.table_size

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

    return net, reservoir, config


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
        reservoir_capacity=args.reservoir_capacity,
        num_equity_rollouts=args.num_equity_rollouts,
        game_config=game_config,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.out_dir, "checkpoint_latest")
    initial_net, initial_reservoir, config = _reload_checkpoint(checkpoint_path, args.reload_previous, config, rng)
    trainer = Trainer.new(config, rng, initial_net=initial_net, initial_reservoir=initial_reservoir)

    eval_fn = None
    if args.eval_interval > 0:
        eval_fn = _make_eval_fn(
            game_config, config.table_size, config.feature_keys, args.eval_rounds, args.eval_genome_path, rng,
        )

    consecutive_non_improvements = 0

    for iteration in range(1, args.iterations + 1):
        # Snapshot the net *before* this iteration's training update -- see
        # --benchmark-interval's help text -- so the benchmark below always
        # compares against exactly what changed this iteration, not some
        # possibly-stale saved checkpoint.
        pre_update_net = None
        if args.benchmark_interval > 0 and iteration % args.benchmark_interval == 0:
            pre_update_net = cfr_networks.clone(trainer.net)

        t0 = time.time()
        mean_loss = run_iteration(trainer, rng, iteration)
        elapsed = time.time() - t0
        print(
            f"iter {iteration:4d} | reservoir {len(trainer.reservoir):7d}/{config.reservoir_capacity} | "
            f"mean loss {mean_loss:10.4f} | {elapsed:5.1f}s"
        )

        if args.feature_importance:
            _print_feature_importance(
                trainer, rng, args.feature_importance_top_n, args.feature_importance_sample_size,
                args.feature_importance_background_size, args.feature_importance_nsamples,
            )

        if args.checkpoint_interval > 0 and iteration % args.checkpoint_interval == 0:
            trainer.save(checkpoint_path)

        if pre_update_net is not None:
            current_players = _benchmark_players(trainer.net, config.feature_keys, "current", id_offset=0)
            pre_update_players = _benchmark_players(pre_update_net, config.feature_keys, "pre_update", id_offset=-100)
            outcome = run_benchmark_until_resolved(
                current_players, pre_update_players, game_config, rng,
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
                f"         | benchmark vs pre-update net | {outcome.tables_played} tables | "
                f"current edge {outcome.mean_bb_per_100:+7.2f} bb/100 "
                f"({confidence_pct:.0f}% CI [{outcome.ci_low:+7.2f}, {outcome.ci_high:+7.2f}]) | {verdict}"
            )
            if outcome.improved:
                consecutive_non_improvements = 0
            else:
                consecutive_non_improvements += 1
                patience_label = args.early_stop_patience if args.early_stop_patience > 0 else "unlimited"
                print(
                    f"         | this update didn't help -- reverting net to its pre-update state "
                    f"({consecutive_non_improvements}/{patience_label} consecutive non-improvements)"
                )
                trainer.revert_to(pre_update_net)
                trainer.save(checkpoint_path)
                if args.early_stop_patience > 0 and consecutive_non_improvements >= args.early_stop_patience:
                    print(
                        f"Early stopping: no improvement for {args.early_stop_patience} "
                        "consecutive benchmark checks."
                    )
                    break

        if eval_fn is not None and args.eval_interval > 0 and iteration % args.eval_interval == 0:
            eval_fn(trainer.net, iteration)

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
