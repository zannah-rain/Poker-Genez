"""CLI: evolve a population of poker genomes over 6-max NLHE sessions.

Example:
    python -m poker_ga.main --generations 50 --population 60 --rounds 3
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from benchmark import run_benchmark
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
    p.add_argument("--generations", type=int, default=30)
    p.add_argument("--population", type=int, default=180, help="Must be a multiple of 6.")
    p.add_argument("--rounds", type=int, default=4, help="Random re-seatings per generation.")
    p.add_argument("--max-hands", type=int, default=500, help="Hand cap per table session.")
    p.add_argument("--starting-stack", type=float, default=200.0)
    p.add_argument("--small-blind", type=float, default=1.0)
    p.add_argument("--big-blind", type=float, default=2.0)
    p.add_argument("--elite", type=int, default=0)
    p.add_argument("--mutation-rate", type=float, default=0.001)
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
        "--final-rounds", type=int, default=200,
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
        help="Population file to reload. Defaults to <final-out-dir>/population.npy.",
    )
    p.add_argument(
        "--benchmark-interval", type=int, default=10,
        help="Every this many generations, play the current population head-to-head against "
        "a saved checkpoint from --benchmark-interval generations ago, in 3-vs-3 tables, and "
        "print aggregate net chips + bb/100 for each side. Unlike the per-generation fitness "
        "number (only comparable against that generation's own random opponents), this is a "
        "direct, apples-to-apples measure of whether evolution is actually improving. Set to "
        "0 to disable.",
    )
    p.add_argument(
        "--benchmark-tables", type=int, default=20,
        help="Number of independent 3-vs-3 tables played for each --benchmark-interval checkpoint match.",
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
    return p.parse_args()


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
    reload_path = args.reload_path or os.path.join(final_out_dir, "population.npy")
    benchmark_dir = os.path.join(args.out_dir, "benchmarks")
    if args.benchmark_interval > 0:
        os.makedirs(benchmark_dir, exist_ok=True)

    seed_genomes = None
    if args.reload_previous:
        if os.path.exists(reload_path):
            try:
                seed_genomes = load_population(reload_path)
                print(f"Reloaded {len(seed_genomes)} genomes from previous run at {reload_path}")
            except Exception as exc:
                print(f"Warning: could not reload population from {reload_path} ({exc}); starting from scratch.")
        else:
            print(f"No previous population found at {reload_path}; starting from scratch.")

    island_model = IslandModel(ga_config, island_config, rng, seed_genomes=seed_genomes)

    for gen in range(args.generations):
        t0 = time.time()
        fitness_by_island = []
        gen_stats_by_island = []
        for island in island_model.islands:
            raw_fitness, gen_stats = run_generation(island.players, game_config, sim_config, rng)
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
        best_player.genome.save(os.path.join(args.out_dir, "best_genome_latest.npy"))
        if gen == args.generations - 1:
            best_player.genome.save(os.path.join(args.out_dir, f"best_genome_gen{gen}.npy"))

        if args.benchmark_interval > 0 and gen % args.benchmark_interval == 0:
            checkpoint_gen = gen - args.benchmark_interval
            checkpoint_path = os.path.join(benchmark_dir, f"gen{checkpoint_gen:05d}_population.npy")
            if checkpoint_gen >= 0 and os.path.exists(checkpoint_path):
                checkpoint_genomes = load_population(checkpoint_path)
                checkpoint_players = [
                    Player(player_id=-(i + 1), genome=g, generation=checkpoint_gen)
                    for i, g in enumerate(checkpoint_genomes)
                ]
                bench = run_benchmark(
                    all_players, checkpoint_players, game_config, rng,
                    num_tables=args.benchmark_tables,
                )
                print(
                    f"         | benchmark vs gen {checkpoint_gen:4d} | "
                    f"current {bench.current_net_total:+9.1f} chips ({bench.bb_per_100('current', args.big_blind):+7.2f} bb/100) "
                    f"| checkpoint {bench.checkpoint_net_total:+9.1f} chips ({bench.bb_per_100('checkpoint', args.big_blind):+7.2f} bb/100)"
                )
            save_population(
                [p.genome for p in all_players],
                os.path.join(benchmark_dir, f"gen{gen:05d}_population.npy"),
            )

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
    final_stats = run_final_tournament(all_players, final_game_config, final_sim_config, rng)
    ranked = rank_players(all_players, final_stats, sparsity_penalty=args.sparsity_penalty)
    export_top_n(ranked, final_stats, final_game_config, args.top_n, final_out_dir)

    population_path = os.path.join(final_out_dir, "population.npy")
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


if __name__ == "__main__":
    main()
