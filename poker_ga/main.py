"""CLI: evolve a population of poker genomes over 6-max NLHE sessions.

Example:
    python -m poker_ga.main --generations 50 --population 60 --rounds 3
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import Executor, ProcessPoolExecutor
from contextlib import nullcontext

import numpy as np

from benchmark import run_benchmark_until_resolved
from ga import GAConfig, IslandConfig, IslandModel
from game import GameConfig
from genome import load_population, save_population
from simulate import SimConfig, combine_generation_stats, run_generation
from player import Player
from tournament import export_top_n, rank_players, run_final_tournament


def apply_sparsity_penalty(players: list[Player], fitness: dict, coefficient: float) -> dict:
    """Subtracts `coefficient` chips per nonzero feature weight from each
    player's fitness, so selection favors sparser (more memorizable)
    genomes alongside raw poker performance."""
    if coefficient <= 0:
        return fitness
    return {
        p.player_id: fitness[p.player_id] - coefficient * p.genome.nonzero_weight_count()
        for p in players
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evolve poker strategies with a genetic algorithm.")
    p.add_argument("--generations", type=int, default=100)
    p.add_argument("--population", type=int, default=180, help="Must be a multiple of 6.")
    p.add_argument("--rounds", type=int, default=100, help="Random re-seatings per generation.")
    p.add_argument("--max-hands", type=int, default=200, help="Hand cap per table session.")
    p.add_argument("--starting-stack", type=float, default=200.0)
    p.add_argument("--small-blind", type=float, default=1.0)
    p.add_argument("--big-blind", type=float, default=2.0)
    p.add_argument("--elite", type=int, default=0)
    p.add_argument("--mutation-rate", type=float, default=0.005)
    p.add_argument("--mutation-scale", type=float, default=0.3)
    p.add_argument(
        "--sparsity-penalty", type=float, default=2.0,
        help="Chips subtracted from fitness per nonzero feature weight (weights_v + "
        "weights_l combined), pushing evolution toward sparser, more memorizable "
        "genomes. Applied both during evolution and to the final tournament ranking. "
        "0 disables it.",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out-dir", type=str, default="runs", help="Where to save the best genome each generation.")
    p.add_argument(
        "--final-rounds", type=int, default=100,
        help="Random re-seatings in the final scoring tournament (more than --rounds for a low-variance ranking).",
    )
    p.add_argument(
        "--final-max-hands", type=int, default=100,
        help="Hand cap per table session in the final scoring tournament.",
    )
    p.add_argument("--top-n", type=int, default=5, help="How many top genomes to export after the final tournament.")
    p.add_argument(
        "--final-out-dir", type=str, default=None,
        help="Where to write the final leaderboard + strategy reports. Defaults to <out-dir>/final.",
    )
    p.add_argument(
        "--reload-previous", action=argparse.BooleanOptionalAction, default=True,
        help="Seed generation 0 from the previous run's final population, if one is found "
        "(pass --no-reload-previous to always start from a fresh random population).",
    )
    p.add_argument(
        "--reload-path", type=str, default=None,
        help="Population file to reload. Defaults to <final-out-dir>/population.json.",
    )
    p.add_argument(
        "--benchmark-interval", type=int, default=10,
        help="Every this many generations, play the current population head-to-head against "
        "a saved checkpoint from --benchmark-interval generations ago, in 3-vs-3 tables, until "
        "the result is statistically resolved (see --benchmark-min/max-tables, "
        "--benchmark-p-value). Unlike the per-generation fitness number (only comparable "
        "against that generation's own random opponents), this is a direct, apples-to-apples "
        "measure of whether evolution is actually improving. Set to 0 to disable.",
    )
    p.add_argument(
        "--benchmark-min-tables", type=int, default=500,
        help="Minimum number of 3-vs-3 tables played against the checkpoint before checking "
        "whether the result is statistically resolved (see --benchmark-p-value). Poker has "
        "enough hand-to-hand variance that fewer tables than this makes the improve/regress "
        "call mostly noise.",
    )
    p.add_argument(
        "--benchmark-max-tables", type=int, default=2000,
        help="Hard cap on 3-vs-3 tables played in one benchmark check. If the confidence "
        "interval still straddles 0 at this point (a genuinely-near-zero edge can take a very "
        "long time to resolve), the check ends anyway and is conservatively treated as 'not "
        "improved'.",
    )
    p.add_argument(
        "--benchmark-table-batch", type=int, default=200,
        help="Additional tables played per round once --benchmark-min-tables isn't enough to "
        "resolve the confidence interval (repeats until it resolves or --benchmark-max-tables "
        "is hit).",
    )
    p.add_argument(
        "--benchmark-p-value", type=float, default=0.1,
        help="Improvement p-value threshold (N): the benchmark keeps playing tables until the "
        "(1 - N) confidence interval of the current population's mean bb/100 edge over the "
        "checkpoint no longer includes 0, i.e. until the direction of the edge is resolved to "
        "this confidence level. Lower values demand stronger evidence before calling a "
        "generation improved or regressed (and so tend to need more tables to resolve).",
    )
    p.add_argument(
        "--early-stop-patience", type=int, default=2,
        help="Whenever a --benchmark-interval check shows the current population hasn't beaten "
        "the previous benchmark checkpoint, training reverts to that checkpoint instead of "
        "continuing from the (worse) current population. If this happens this many times in a "
        "row, training stops early rather than running out --generations. Set to 0 to never "
        "stop early (still reverts on non-improvement, just keeps retrying indefinitely). Only "
        "applies when --benchmark-interval > 0.",
    )
    p.add_argument(
        "--num-islands", type=int, default=3,
        help="Splits --population into this many independent islands, each with its own "
        "breeding pool AND its own tables (an island's players only ever face other members "
        "of that island -- see ga.py's IslandModel). Keeps genetic diversity alive: a "
        "pathological strategy that takes over one island's fitness landscape doesn't "
        "automatically spread to the others, the way it would in one shared population. Must "
        "evenly divide --population into groups that are each a multiple of 6. Set to 1 to "
        "disable (behaves like a single population, as before).",
    )
    p.add_argument(
        "--migration-interval", type=int, default=10,
        help="Every this many generations, each island's best --migration-size genomes are "
        "copied into the next island in a ring, replacing random non-elite slots there. The "
        "only channel connecting islands -- lets good strategies spread (or rescue a "
        "collapsed island) without homogenizing everything immediately. 0 disables migration. "
        "Ignored if --num-islands is 1.",
    )
    p.add_argument(
        "--migration-size", type=int, default=3,
        help="How many of an island's best genomes migrate to the next island per migration event.",
    )
    p.add_argument(
        "--workers", type=int, default=0,
        help="Number of worker processes to spread table matches across (used for the "
        "per-generation fitness pass, the checkpoint benchmark, and the final tournament). "
        "1 is fully sequential -- identical to this project's original "
        "single-process behavior, including exact reproducibility for a given --seed. Table "
        "matches are fully independent of each other, so this parallelizes almost for free, "
        "but a parallel run necessarily consumes randomness differently than a sequential "
        "one (each table needs its own independent random stream rather than fighting over "
        "one shared one), so --workers > 1 will not reproduce the exact same hands as "
        "--workers 1 for the same --seed -- only the same results run-to-run for a fixed "
        "(--seed, --workers) pair. 0 or a negative number means 'use every available CPU "
        "core'.",
    )
    return p.parse_args()


def _run_training(args: argparse.Namespace, rng: np.random.Generator, num_workers: int, executor: Executor | None) -> None:
    island_config = IslandConfig(
        num_islands=args.num_islands,
        migration_interval=args.migration_interval,
        migration_size=args.migration_size,
    )
    ga_config = GAConfig(
        population_size=args.population // args.num_islands,  # per-island size
        elite_count=args.elite,
        mutation_rate=args.mutation_rate,
        mutation_scale=args.mutation_scale,
    )
    game_config = GameConfig(
        small_blind=args.small_blind,
        big_blind=args.big_blind,
        starting_stack=args.starting_stack,
        max_hands_per_session=args.max_hands,
    )
    sim_config = SimConfig(rounds_per_generation=args.rounds)

    os.makedirs(args.out_dir, exist_ok=True)
    final_out_dir = args.final_out_dir or os.path.join(args.out_dir, "final")
    # A single, continuously-overwritten snapshot of the full population,
    # saved every generation (not one file per generation, to avoid
    # bloating disk on long runs) -- this is what lets a run resume from
    # wherever it last got to, rather than only from a fully completed
    # run's final tournament output.
    latest_population_path = os.path.join(args.out_dir, "latest_population.json")
    final_population_path = os.path.join(final_out_dir, "population.json")
    default_reload_path = latest_population_path if os.path.exists(latest_population_path) else final_population_path
    reload_path = args.reload_path or default_reload_path

    benchmark_dir = os.path.join(args.out_dir, "benchmarks")
    # Also a single, continuously-overwritten snapshot -- but only advanced
    # when a benchmark check confirms improvement (see the training loop
    # below), so it always holds the last population that was actually
    # measured to be better than the one before it.
    benchmark_checkpoint_path = os.path.join(benchmark_dir, "checkpoint_population.json")
    if args.benchmark_interval > 0:
        os.makedirs(benchmark_dir, exist_ok=True)

    seed_genomes = None
    if args.reload_previous:
        if os.path.exists(reload_path):
            try:
                seed_genomes = load_population(reload_path, rng)
                print(f"Reloaded {len(seed_genomes)} genomes from previous run at {reload_path}")
            except Exception as exc:
                print(f"Warning: could not reload population from {reload_path} ({exc}); starting from scratch.")
        else:
            print(f"No previous population found at {reload_path}; starting from scratch.")

    island_model = IslandModel(ga_config, island_config, rng, seed_genomes=seed_genomes)
    consecutive_non_improvements = 0

    for gen in range(args.generations):
        t0 = time.time()
        fitness_by_island = []
        gen_stats_by_island = []
        for i, island in enumerate(island_model.islands):
            desc = f"gen {gen} eval (island {i + 1}/{len(island_model.islands)})" if len(island_model.islands) > 1 \
                else f"gen {gen} eval"
            raw_fitness, gen_stats = run_generation(
                island.players, game_config, sim_config, rng, progress_desc=desc,
                num_workers=num_workers, executor=executor,
            )
            fitness_by_island.append(apply_sparsity_penalty(island.players, raw_fitness, args.sparsity_penalty))
            gen_stats_by_island.append(gen_stats)

        all_players = island_model.all_players
        all_fitness = {pid: v for fitness in fitness_by_island for pid, v in fitness.items()}
        values = np.array(list(all_fitness.values()))
        nonzero_counts = np.array([p.genome.nonzero_weight_count() for p in all_players])
        best_player = max(all_players, key=lambda p: all_fitness[p.player_id])
        combined_stats = combine_generation_stats(gen_stats_by_island)
        elapsed = time.time() - t0

        print(
            f"gen {gen:4d} | best {values.max() / sim_config.rounds_per_generation:9.1f} | mean {values.mean() / sim_config.rounds_per_generation:8.1f} "
            f"| worst {values.min() / sim_config.rounds_per_generation:9.1f} | std {values.std() / sim_config.rounds_per_generation:7.1f} "
            f"| nonzero wts avg {nonzero_counts.mean():5.1f} min {nonzero_counts.min():3d} "
            f"| {elapsed:5.1f}s"
        )
        print(
            f"         | sense check: mean hands survived {combined_stats.mean_hands_survived:6.1f} "
            f"| bust rate {combined_stats.bust_rate:6.1%} | fold rate facing bet {combined_stats.fold_rate_facing_bet:6.1%} "
            f"| mean raises/street {combined_stats.mean_raises_per_street:4.2f}"
        )
        if len(island_model.islands) > 1:
            per_island = " | ".join(
                f"isl{i} best {max(fitness_by_island[i].values()) / sim_config.rounds_per_generation:7.1f} "
                f"fold {gen_stats_by_island[i].fold_rate_facing_bet:5.1%}"
                for i in range(len(island_model.islands))
            )
            print(f"         | islands: {per_island}")
        best_player.genome.save(os.path.join(args.out_dir, "best_genome_latest.json"))
        if gen == args.generations - 1:
            best_player.genome.save(os.path.join(args.out_dir, f"best_genome_gen{gen}.json"))

        # Overwrite the resumable "latest" snapshot every generation (not
        # once per generation as a separate file, to avoid bloating disk on
        # long runs) so a killed/interrupted run can pick back up here.
        save_population([p.genome for p in all_players], latest_population_path)

        reverted = False
        if args.benchmark_interval > 0 and gen % args.benchmark_interval == 0:
            if os.path.exists(benchmark_checkpoint_path):
                checkpoint_genomes = load_population(benchmark_checkpoint_path, rng)
                checkpoint_players = [
                    Player(player_id=-(i + 1), genome=g, generation=gen - args.benchmark_interval)
                    for i, g in enumerate(checkpoint_genomes)
                ]
                outcome = run_benchmark_until_resolved(
                    all_players, checkpoint_players, game_config, rng,
                    min_tables=args.benchmark_min_tables,
                    max_tables=args.benchmark_max_tables,
                    table_batch=args.benchmark_table_batch,
                    p_value=args.benchmark_p_value,
                    num_workers=num_workers, executor=executor,
                )
                if outcome.resolved:
                    verdict = "IMPROVED" if outcome.improved else "REGRESSED"
                else:
                    verdict = "INCONCLUSIVE (table cap hit; treated as not improved)"
                confidence_pct = (1.0 - args.benchmark_p_value) * 100.0
                print(
                    f"         | benchmark vs checkpoint | {outcome.tables_played} tables | "
                    f"current edge {outcome.mean_bb_per_100:+7.2f} bb/100 "
                    f"({confidence_pct:.0f}% CI [{outcome.ci_low:+7.2f}, {outcome.ci_high:+7.2f}]) | {verdict}"
                )
                improved = outcome.improved
                if improved:
                    consecutive_non_improvements = 0
                    save_population([p.genome for p in all_players], benchmark_checkpoint_path)
                else:
                    consecutive_non_improvements += 1
                    print(
                        f"         | reverting to benchmark checkpoint "
                        f"({consecutive_non_improvements}/{args.early_stop_patience or '∞'} consecutive non-improvements)"
                    )
                    island_model = IslandModel(ga_config, island_config, rng, seed_genomes=checkpoint_genomes)
                    save_population([p.genome for p in island_model.all_players], latest_population_path)
                    reverted = True
                    if args.early_stop_patience > 0 and consecutive_non_improvements >= args.early_stop_patience:
                        print(
                            f"Early stopping: no improvement for {args.early_stop_patience} "
                            "consecutive benchmark checks."
                        )
                        break
            else:
                # First benchmark interval -- nothing to compare against yet,
                # so just bootstrap the checkpoint from the current population.
                save_population([p.genome for p in all_players], benchmark_checkpoint_path)

        if not reverted:
            island_model.evolve_all(fitness_by_island)

    all_players = island_model.all_players
    print(
        f"\nRunning final tournament: {len(all_players)} genomes from generation "
        f"{island_model.generation}, {args.final_rounds} rounds, up to {args.final_max_hands} hands/session..."
    )
    final_sim_config = SimConfig(rounds_per_generation=args.final_rounds)
    final_game_config = GameConfig(
        small_blind=args.small_blind,
        big_blind=args.big_blind,
        starting_stack=args.starting_stack,
        max_hands_per_session=args.final_max_hands,
    )
    t0 = time.time()
    final_stats = run_final_tournament(
        all_players, final_game_config, final_sim_config, rng,
        num_workers=num_workers, executor=executor,
    )
    ranked = rank_players(all_players, final_stats, sparsity_penalty=args.sparsity_penalty)
    export_top_n(ranked, final_stats, final_game_config, args.top_n, final_out_dir)

    population_path = os.path.join(final_out_dir, "population.json")
    save_population([p.genome for p in ranked], population_path)
    elapsed = time.time() - t0

    print(f"Final tournament complete in {elapsed:.1f}s. Top {args.top_n}:")
    for rank, p in enumerate(ranked[: args.top_n], start=1):
        s = final_stats[p.player_id]
        name = p.label or f"Player {p.player_id}"
        print(
            f"  #{rank} {name:12s} | mean {s.mean_net_chips:+8.1f}/session | "
            f"win {s.win_rate:6.1%} | bust {s.bust_rate:6.1%} | {s.bb_per_100(args.big_blind):+7.2f} bb/100 "
            f"| {p.genome.nonzero_weight_count():3d} nonzero wts"
        )
    print(f"Strategy reports written to {final_out_dir}/")
    print(f"Full ranked population ({len(ranked)} genomes) saved to {population_path} for --reload-previous.")


def main() -> None:
    args = parse_args()
    if args.num_islands < 1:
        raise SystemExit("--num-islands must be at least 1.")
    if args.population % (args.num_islands * 6) != 0:
        raise SystemExit(
            f"--population must be a multiple of --num-islands * 6 "
            f"({args.num_islands * 6}) for clean 6-max seating within each island."
        )

    rng = np.random.default_rng(args.seed)
    num_workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)

    # A single, long-lived worker pool reused for the whole run (per-generation
    # fitness passes across every island, benchmark checks, and the final
    # tournament) rather than one spun up and torn down per call -- process
    # start-up has real, repeated cost, so amortizing it over the entire run
    # matters far more than over any single call. Sequential runs (the
    # default, num_workers == 1) skip pool creation entirely.
    pool_context = ProcessPoolExecutor(max_workers=num_workers) if num_workers > 1 else nullcontext(None)
    with pool_context as executor:
        _run_training(args, rng, num_workers, executor)


if __name__ == "__main__":
    main()
