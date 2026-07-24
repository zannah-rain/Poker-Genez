import os

import numpy as np
import pytest

import main as main_module
from features import NUM_FEATURES
from genome import Genome, quantize
from main import apply_sparsity_penalty, main, parse_args
from player import Player


def make_genome_with_nonzero_count(count):
    weights_v = quantize(np.zeros(NUM_FEATURES))
    for i in range(count):
        weights_v[i] = 10.0
    from gto import NUM_GTO_SPOTS
    return Genome(
        weights_v=weights_v, weights_l=quantize(np.zeros(NUM_FEATURES)),
        bias_v=50.0, bias_l=50.0, theta_value=70.0, theta_bluff=70.0, theta_call=40.0,
        kappa=0.5, noise_std=1.0, gto_flags=np.zeros(NUM_GTO_SPOTS),
    )


class TestParseArgs:
    def test_defaults(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["main.py"])
        args = parse_args()
        assert args.generations == 100
        assert args.population == 180
        assert args.sparsity_penalty == 2.0
        assert args.num_islands == 3
        assert args.reload_previous is True

    def test_overrides_are_applied(self, monkeypatch):
        monkeypatch.setattr("sys.argv", [
            "main.py", "--generations", "5", "--population", "30",
            "--sparsity-penalty", "0", "--num-islands", "1",
        ])
        args = parse_args()
        assert args.generations == 5
        assert args.population == 30
        assert args.sparsity_penalty == 0.0
        assert args.num_islands == 1

    def test_no_reload_previous_flag(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["main.py", "--no-reload-previous"])
        args = parse_args()
        assert args.reload_previous is False


class TestApplySparsityPenalty:
    def test_zero_coefficient_returns_fitness_unchanged(self):
        players = [Player(player_id=1, genome=make_genome_with_nonzero_count(5))]
        fitness = {1: 100.0}
        result = apply_sparsity_penalty(players, fitness, coefficient=0.0)
        assert result == fitness

    def test_negative_coefficient_returns_fitness_unchanged(self):
        players = [Player(player_id=1, genome=make_genome_with_nonzero_count(5))]
        fitness = {1: 100.0}
        result = apply_sparsity_penalty(players, fitness, coefficient=-1.0)
        assert result == fitness

    def test_positive_coefficient_subtracts_per_nonzero_weight(self):
        players = [
            Player(player_id=1, genome=make_genome_with_nonzero_count(0)),
            Player(player_id=2, genome=make_genome_with_nonzero_count(10)),
        ]
        fitness = {1: 100.0, 2: 100.0}
        result = apply_sparsity_penalty(players, fitness, coefficient=2.0)
        assert result[1] == 100.0
        assert result[2] == 100.0 - 2.0 * 10


class TestMainSmoke:
    def test_main_runs_end_to_end_on_a_tiny_configuration(self, tmp_path, monkeypatch, capsys):
        out_dir = str(tmp_path / "runs")
        monkeypatch.setattr("sys.argv", [
            "main.py",
            "--generations", "1",
            "--population", "6",
            "--rounds", "1",
            "--max-hands", "3",
            "--final-rounds", "1",
            "--final-max-hands", "2",
            "--num-islands", "1",
            "--benchmark-interval", "0",
            "--top-n", "1",
            "--out-dir", out_dir,
            "--no-reload-previous",
            "--seed", "0",
        ])
        main()
        assert os.path.exists(os.path.join(out_dir, "best_genome_latest.json"))
        assert os.path.exists(os.path.join(out_dir, "latest_population.json"))
        final_dir = os.path.join(out_dir, "final")
        assert os.path.exists(os.path.join(final_dir, "leaderboard.md"))
        assert os.path.exists(os.path.join(final_dir, "population.json"))
