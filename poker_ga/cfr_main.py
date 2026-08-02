"""CLI: train a Single Deep CFR advantage network over 6-max NLHE.

Example:
    python -m poker_ga.cfr_main --iterations 200 --traversals-per-iteration 200 --table-size 6
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import cfr_features
import cfr_policy
import cfr_tree
from cfr_train import DeepCFRConfig, train
from game import GameConfig
from genome import Genome
from player import Player
from simulate import SimConfig
from tournament import run_final_tournament


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Single Deep CFR poker strategy.")
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--traversals-per-iteration", type=int, default=200)
    p.add_argument("--sgd-steps-per-iteration", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--reservoir-capacity", type=int, default=200_000)
    p.add_argument(
        "--num-equity-rollouts", type=int, default=cfr_tree.DEFAULT_NUM_EQUITY_ROLLOUTS,
        help="Terminal showdowns reached before the river (an early all-in) average this many "
        "possible board completions instead of dealing one random runout, as a lower-variance "
        "regression target -- exact whenever that many completions covers every possibility "
        "(e.g. a river-only completion), a Monte Carlo estimate otherwise (e.g. a preflop all-in).",
    )
    p.add_argument(
        "--hidden-sizes", type=str, default="128,128",
        help="Comma-separated advantage-net hidden layer sizes.",
    )
    p.add_argument("--table-size", type=int, default=6)
    p.add_argument(
        "--feature-keys", type=str, default=None,
        help="Comma-separated features.FEATURE_NAMES keys the net conditions on. "
        "Defaults to cfr_features.DEFAULT_FEATURE_KEYS (~49 generalized features, "
        "excluding opponent-tendency reads -- see cfr_features.py).",
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
        "--eval-interval", type=int, default=0,
        help="Every this many iterations, benchmark the current net (bb/100) against a table of "
        "freshly random GA genomes (genome.Genome.random) via tournament.run_final_tournament. "
        "0 disables eval.",
    )
    p.add_argument("--eval-rounds", type=int, default=20, help="Rounds of re-seating for each --eval-interval check.")
    return p.parse_args()


def _make_eval_fn(game_config: GameConfig, table_size: int, feature_keys: tuple[str, ...], eval_rounds: int, rng: np.random.Generator):
    def eval_fn(net, iteration: int) -> None:
        cfr_player = Player(player_id=0, genome=cfr_policy.DeepCFRPolicy(net, feature_keys), label="DeepCFR")
        baseline_players = [
            Player(player_id=i, genome=Genome.random(rng), label=f"random_ga_{i}")
            for i in range(1, table_size)
        ]
        sim_config = SimConfig(rounds_per_generation=eval_rounds, table_size=table_size)
        stats = run_final_tournament(
            [cfr_player, *baseline_players], game_config, sim_config, rng, show_progress=False,
        )
        bb_per_100 = stats[cfr_player.player_id].bb_per_100(game_config.big_blind)
        print(f"         | eval @ iter {iteration}: DeepCFR vs random GA genomes = {bb_per_100:+.2f} bb/100")

    return eval_fn


def main() -> None:
    args = parse_args()
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

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    eval_fn = None
    if args.eval_interval > 0:
        eval_fn = _make_eval_fn(game_config, args.table_size, feature_keys, args.eval_rounds, rng)

    train(
        config, rng, args.out_dir,
        checkpoint_interval=args.checkpoint_interval,
        eval_interval=args.eval_interval,
        eval_fn=eval_fn,
    )
    print(f"Done. Checkpoint saved to {os.path.join(args.out_dir, 'checkpoint_latest')}.{{pt,json}}")


if __name__ == "__main__":
    main()
