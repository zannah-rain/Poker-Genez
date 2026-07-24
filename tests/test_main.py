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

    def test_benchmark_defaults(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["main.py"])
        args = parse_args()
        assert args.benchmark_min_tables == 100
        assert args.benchmark_max_tables == 1000
        assert args.benchmark_table_batch == 100
        assert args.benchmark_p_value == 0.1

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

    def test_benchmark_overrides_are_applied(self, monkeypatch):
        monkeypatch.setattr("sys.argv", [
            "main.py",
            "--benchmark-min-tables", "50",
            "--benchmark-max-tables", "500",
            "--benchmark-table-batch", "25",
            "--benchmark-p-value", "0.01",
        ])
        args = parse_args()
        assert args.benchmark_min_tables == 50
        assert args.benchmark_max_tables == 500
        assert args.benchmark_table_batch == 25
        assert args.benchmark_p_value == 0.01

    def test_no_reload_previous_flag(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["main.py", "--no-reload-previous"])
        args = parse_args()
        assert args.reload_previous is False

    def test_workers_defaults_to_sequential(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["main.py"])
        args = parse_args()
        assert args.workers == 1

    def test_workers_override(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["main.py", "--workers", "4"])
        args = parse_args()
        assert args.workers == 4


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

    def test_main_exercises_the_benchmark_and_checkpoint_path(self, tmp_path, monkeypatch, capsys):
        out_dir = str(tmp_path / "runs")
        monkeypatch.setattr("sys.argv", [
            "main.py",
            "--generations", "2",
            "--population", "6",
            "--rounds", "1",
            "--max-hands", "3",
            "--final-rounds", "1",
            "--final-max-hands", "2",
            "--num-islands", "1",
            "--benchmark-interval", "1",
            "--benchmark-min-tables", "4",
            "--benchmark-max-tables", "4",
            "--benchmark-table-batch", "4",
            "--top-n", "1",
            "--out-dir", out_dir,
            "--no-reload-previous",
            "--seed", "0",
        ])
        main()
        checkpoint_path = os.path.join(out_dir, "benchmarks", "checkpoint_population.json")
        assert os.path.exists(checkpoint_path)
        output = capsys.readouterr().out
        assert "benchmark vs checkpoint" in output

    def test_main_runs_with_multiple_workers(self, tmp_path, monkeypatch):
        # Exercises the real ProcessPoolExecutor path: pool creation in
        # main(), reuse across islands/generations/the final tournament, and
        # teardown at the end.
        out_dir = str(tmp_path / "runs")
        monkeypatch.setattr("sys.argv", [
            "main.py",
            "--generations", "1",
            "--population", "12",
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
            "--workers", "2",
        ])
        main()
        assert os.path.exists(os.path.join(out_dir, "best_genome_latest.json"))
        final_dir = os.path.join(out_dir, "final")
        assert os.path.exists(os.path.join(final_dir, "leaderboard.md"))
