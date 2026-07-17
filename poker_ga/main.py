"""CLI: evolve a population of poker genomes over 6-max NLHE sessions.

Example:
    python -m poker_ga.main --generations 50 --population 60 --rounds 3
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from ga import GAConfig, Population
from game import GameConfig
from genome import load_population, save_population
from simulate import SimConfig, run_generation
from tournament import export_top_n, rank_players, run_final_tournament


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evolve poker strategies with a genetic algorithm.")
    p.add_argument("--generations", type=int, default=10)
    p.add_argument("--population", type=int, default=120, help="Must be a multiple of 6.")
    p.add_argument("--rounds", type=int, default=25, help="Random re-seatings per generation.")
    p.add_argument("--max-hands", type=int, default=50, help="Hand cap per table session.")
    p.add_argument(
        "--busts-before-table-ends", type=int, default=2,
        help="End a table's session once this many of its original players have busted, "
        "rather than always playing down to heads-up. Models real tables refilling empty "
        "seats with new players, and keeps the GA from over-adapting to short-handed "
        "end-games it won't mostly face.",
    )
    p.add_argument("--starting-stack", type=float, default=200.0)
    p.add_argument("--small-blind", type=float, default=1.0)
    p.add_argument("--big-blind", type=float, default=2.0)
    p.add_argument("--elite", type=int, default=4)
    p.add_argument("--mutation-rate", type=float, default=0.15)
    p.add_argument("--mutation-scale", type=float, default=0.3)
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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.population % 6 != 0:
        raise SystemExit("--population must be a multiple of 6 for clean 6-max seating.")
    if args.busts_before_table_ends < 1:
        raise SystemExit("--busts-before-table-ends must be at least 1.")

    rng = np.random.default_rng(args.seed)

    ga_config = GAConfig(
        population_size=args.population,
        elite_count=args.elite,
        mutation_rate=args.mutation_rate,
        mutation_scale=args.mutation_scale,
    )
    game_config = GameConfig(
        small_blind=args.small_blind,
        big_blind=args.big_blind,
        starting_stack=args.starting_stack,
        max_hands_per_session=args.max_hands,
        busts_before_table_ends=args.busts_before_table_ends,
    )
    sim_config = SimConfig(rounds_per_generation=args.rounds)

    os.makedirs(args.out_dir, exist_ok=True)
    final_out_dir = args.final_out_dir or os.path.join(args.out_dir, "final")
    reload_path = args.reload_path or os.path.join(final_out_dir, "population.npy")

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

    population = Population(ga_config, rng, seed_genomes=seed_genomes)

    for gen in range(args.generations):
        t0 = time.time()
        fitness = run_generation(population.players, game_config, sim_config, rng)
        values = np.array(list(fitness.values()))
        best_player = max(population.players, key=lambda p: fitness[p.player_id])
        elapsed = time.time() - t0

        print(
            f"gen {gen:4d} | best {values.max():9.1f} | mean {values.mean():8.1f} "
            f"| worst {values.min():9.1f} | std {values.std():7.1f} | {elapsed:5.1f}s"
        )
        best_player.genome.save(os.path.join(args.out_dir, "best_genome_latest.npy"))
        if gen == args.generations - 1:
            best_player.genome.save(os.path.join(args.out_dir, f"best_genome_gen{gen}.npy"))

        population.evolve(fitness)

    print(
        f"\nRunning final tournament: {len(population.players)} genomes from generation "
        f"{population.generation}, {args.final_rounds} rounds, up to {args.final_max_hands} hands/session..."
    )
    final_sim_config = SimConfig(rounds_per_generation=args.final_rounds)
    final_game_config = GameConfig(
        small_blind=args.small_blind,
        big_blind=args.big_blind,
        starting_stack=args.starting_stack,
        max_hands_per_session=args.final_max_hands,
        busts_before_table_ends=args.busts_before_table_ends,
    )
    t0 = time.time()
    final_stats = run_final_tournament(population.players, final_game_config, final_sim_config, rng)
    ranked = rank_players(population.players, final_stats)
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
            f"win {s.win_rate:6.1%} | bust {s.bust_rate:6.1%} | {s.bb_per_100(args.big_blind):+7.2f} bb/100"
        )
    print(f"Strategy reports written to {final_out_dir}/")
    print(f"Full ranked population ({len(ranked)} genomes) saved to {population_path} for --reload-previous.")


if __name__ == "__main__":
    main()
