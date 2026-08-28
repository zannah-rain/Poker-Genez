import os

import numpy as np
import pytest
import torch

import cfr_networks
from cfr_main import _load_benchmark_pool, _reload_checkpoint, _save_benchmark_pool, main, parse_args
from cfr_train import DeepCFRConfig, Trainer
from game import GameConfig

_FEATURE_KEYS = "hand_category_norm,street_norm,call_amount_norm,raises_preflop_norm"

_TINY_TRAINING_ARGS = [
    "--traversals-per-iteration", "2",
    "--sgd-steps-per-iteration", "1",
    "--batch-size", "8",
    "--reservoir-capacity", "50",
    "--num-equity-rollouts", "1",
    "--table-size", "2",
    "--hidden-sizes", "8",
    "--feature-keys", _FEATURE_KEYS,
    "--checkpoint-interval", "1",
    "--benchmark-interval", "0",
    "--workers", "1",
    "--seed", "0",
]


class TestReloadCheckpoint:
    def test_no_reload_previous_returns_nothing_to_resume_from(self, tmp_path):
        config = DeepCFRConfig(feature_keys=("street_norm",), hidden_sizes=(8,), table_size=2)
        net, reservoir, optimizer_state, completed_iterations, out_config = _reload_checkpoint(
            str(tmp_path / "checkpoint_latest"), False, config, np.random.default_rng(0),
        )
        assert net is None
        assert reservoir is None
        assert optimizer_state is None
        assert completed_iterations == 0
        assert out_config is config

    def test_no_checkpoint_on_disk_returns_nothing_to_resume_from(self, tmp_path):
        config = DeepCFRConfig(feature_keys=("street_norm",), hidden_sizes=(8,), table_size=2)
        net, reservoir, optimizer_state, completed_iterations, out_config = _reload_checkpoint(
            str(tmp_path / "checkpoint_latest"), True, config, np.random.default_rng(0),
        )
        assert net is None
        assert reservoir is None
        assert optimizer_state is None
        assert completed_iterations == 0

    def test_reloads_completed_iterations_and_optimizer_state_from_a_saved_trainer(self, tmp_path):
        feature_keys = ("hand_category_norm", "street_norm")
        config = DeepCFRConfig(feature_keys=feature_keys, hidden_sizes=(8,), table_size=2, lr=1e-3)
        trainer = Trainer.new(config, np.random.default_rng(0))
        trainer.completed_iterations = 17
        checkpoint_path = str(tmp_path / "checkpoint_latest")
        trainer.save(checkpoint_path)

        net, reservoir, optimizer_state, completed_iterations, out_config = _reload_checkpoint(
            checkpoint_path, True, DeepCFRConfig(feature_keys=feature_keys, hidden_sizes=(8,), table_size=2),
            np.random.default_rng(0),
        )
        assert net is not None
        assert reservoir is not None
        assert optimizer_state is not None
        assert completed_iterations == 17

    def test_checkpoint_without_a_trainer_state_file_falls_back_to_zero(self, tmp_path):
        # Simulates a checkpoint saved before optimizer/iteration
        # persistence existed: net + reservoir on disk, no
        # `_trainer_state.pt` sidecar.
        feature_keys = ("hand_category_norm", "street_norm")
        config = DeepCFRConfig(feature_keys=feature_keys, hidden_sizes=(8,), table_size=2)
        trainer = Trainer.new(config, np.random.default_rng(0))
        checkpoint_path = str(tmp_path / "checkpoint_latest")
        trainer.save(checkpoint_path)
        os.remove(f"{checkpoint_path}_trainer_state.pt")

        net, reservoir, optimizer_state, completed_iterations, out_config = _reload_checkpoint(
            checkpoint_path, True, DeepCFRConfig(feature_keys=feature_keys, hidden_sizes=(8,), table_size=2),
            np.random.default_rng(0),
        )
        assert net is not None
        assert optimizer_state is None
        assert completed_iterations == 0

    def test_requested_capacity_larger_than_reloaded_reservoir_grows_it(self, tmp_path):
        # Regression coverage: requesting *more* --reservoir-capacity than
        # a reloaded reservoir was saved with should grow that reservoir to
        # match (see ReservoirBuffer.grow), not silently discard the extra
        # requested capacity by falling back to the smaller saved one.
        feature_keys = ("hand_category_norm", "street_norm")
        small_config = DeepCFRConfig(feature_keys=feature_keys, hidden_sizes=(8,), table_size=2, reservoir_capacity=10)
        trainer = Trainer.new(small_config, np.random.default_rng(0))
        checkpoint_path = str(tmp_path / "checkpoint_latest")
        trainer.save(checkpoint_path)

        requested_config = DeepCFRConfig(
            feature_keys=feature_keys, hidden_sizes=(8,), table_size=2, reservoir_capacity=50,
        )
        net, reservoir, optimizer_state, completed_iterations, out_config = _reload_checkpoint(
            checkpoint_path, True, requested_config, np.random.default_rng(0),
        )
        assert reservoir.capacity == 50
        assert out_config.reservoir_capacity == 50

    def test_requested_capacity_smaller_than_reloaded_reservoir_keeps_the_larger_one(self, tmp_path):
        # The other direction is unchanged: shrinking would mean discarding
        # already-collected samples, so the reloaded reservoir's own
        # (larger) capacity wins instead of the smaller request.
        feature_keys = ("hand_category_norm", "street_norm")
        large_config = DeepCFRConfig(feature_keys=feature_keys, hidden_sizes=(8,), table_size=2, reservoir_capacity=50)
        trainer = Trainer.new(large_config, np.random.default_rng(0))
        checkpoint_path = str(tmp_path / "checkpoint_latest")
        trainer.save(checkpoint_path)

        requested_config = DeepCFRConfig(
            feature_keys=feature_keys, hidden_sizes=(8,), table_size=2, reservoir_capacity=10,
        )
        net, reservoir, optimizer_state, completed_iterations, out_config = _reload_checkpoint(
            checkpoint_path, True, requested_config, np.random.default_rng(0),
        )
        assert reservoir.capacity == 50
        assert out_config.reservoir_capacity == 50


class TestResumeContinuesIterationNumbering:
    def test_resumed_run_continues_the_iteration_count_instead_of_restarting_at_one(self, tmp_path, monkeypatch, capsys):
        out_dir = str(tmp_path / "cfr_runs")

        monkeypatch.setattr("sys.argv", [
            "cfr_main.py", "--iterations", "2", "--out-dir", out_dir,
            "--no-reload-previous", *_TINY_TRAINING_ARGS,
        ])
        main()
        capsys.readouterr()  # drain the first run's output; only the resumed run's is under test below
        checkpoint_path = os.path.join(out_dir, "checkpoint_latest")
        assert os.path.exists(f"{checkpoint_path}_trainer_state.pt")
        _, completed_after_first_run = Trainer.load_trainer_state(checkpoint_path)
        assert completed_after_first_run == 2

        monkeypatch.setattr("sys.argv", [
            "cfr_main.py", "--iterations", "2", "--out-dir", out_dir, *_TINY_TRAINING_ARGS,
        ])
        main()
        output = capsys.readouterr().out
        assert "iter    1 " not in output
        assert "iter    3 " in output
        assert "iter    4 " in output
        _, completed_after_second_run = Trainer.load_trainer_state(checkpoint_path)
        assert completed_after_second_run == 4


def _tiny_args_with(**overrides) -> list[str]:
    """_TINY_TRAINING_ARGS with one or more of its own flag values replaced
    (e.g. _tiny_args_with(benchmark_interval=100) -> ..., "--benchmark-interval",
    "100", ...) -- safer than filtering the flat list by value, since more
    than one flag there (--benchmark-interval and --seed) happens to share
    the value "0"."""
    args = list(_TINY_TRAINING_ARGS)
    for flag, value in overrides.items():
        idx = args.index(f"--{flag.replace('_', '-')}")
        args[idx + 1] = str(value)
    return args


def _nets_have_equal_weights(a, b) -> bool:
    a_state, b_state = a.state_dict(), b.state_dict()
    return a_state.keys() == b_state.keys() and all(torch.equal(a_state[k], b_state[k]) for k in a_state)


class TestBenchmarkPoolPersistence:
    """_save_benchmark_pool/_load_benchmark_pool round-trip the rotating
    --benchmark-pool-size pool (see cfr_main.py's own module docstring) so
    a resumed run picks its pool up exactly where a previous run of the
    same checkpoint left it, rather than silently collapsing back to a
    single entry -- "a resumed run must not differ from one that simply
    kept running in the same process" applies to the pool just as much as
    it does to the net/reservoir/optimizer state _reload_checkpoint
    already covers above."""

    def _config(self):
        # Matches _TINY_TRAINING_ARGS' own --feature-keys/--hidden-sizes/
        # --table-size exactly -- test_resumed_run_reloads_the_full_pool_
        # instead_of_collapsing_to_one below feeds this config's own nets
        # into a real _TINY_TRAINING_ARGS run via main(), which would
        # otherwise fail to reload them (a real architecture mismatch, not
        # a test bug to work around).
        return DeepCFRConfig(feature_keys=tuple(_FEATURE_KEYS.split(",")), hidden_sizes=(8,), table_size=2)

    def test_round_trips_every_member_in_order_with_identical_weights(self, tmp_path):
        config = self._config()
        rng = np.random.default_rng(0)
        net_a = Trainer.new(config, rng).net
        net_b = Trainer.new(config, rng).net  # a fresh random init -- different weights from net_a
        assert not _nets_have_equal_weights(net_a, net_b)
        checkpoint_path = str(tmp_path / "checkpoint_latest")
        net_config = cfr_networks.AdvantageNetConfig(
            feature_keys=config.feature_keys, hidden_sizes=config.hidden_sizes, table_size=config.table_size,
        )

        _save_benchmark_pool([net_a, net_b], net_config, checkpoint_path)
        reloaded = _load_benchmark_pool(checkpoint_path)

        assert len(reloaded) == 2
        assert _nets_have_equal_weights(reloaded[0], net_a)  # oldest first
        assert _nets_have_equal_weights(reloaded[1], net_b)

    def test_no_saved_pool_returns_empty_list(self, tmp_path):
        assert _load_benchmark_pool(str(tmp_path / "checkpoint_latest")) == []

    def test_shrinking_the_pool_removes_stale_members_from_disk(self, tmp_path):
        # A later save with fewer members (e.g. --benchmark-pool-size was
        # reduced between runs) must not leave the old, now-orphaned
        # member files behind -- _load_benchmark_pool would otherwise
        # resurrect them on the next resume even though they're no longer
        # part of the pool.
        config = self._config()
        rng = np.random.default_rng(0)
        nets = [Trainer.new(config, rng).net for _ in range(3)]
        checkpoint_path = str(tmp_path / "checkpoint_latest")
        net_config = cfr_networks.AdvantageNetConfig(
            feature_keys=config.feature_keys, hidden_sizes=config.hidden_sizes, table_size=config.table_size,
        )

        _save_benchmark_pool(nets, net_config, checkpoint_path)
        assert len(_load_benchmark_pool(checkpoint_path)) == 3

        _save_benchmark_pool(nets[:1], net_config, checkpoint_path)
        assert len(_load_benchmark_pool(checkpoint_path)) == 1

    def test_resumed_run_reloads_the_full_pool_instead_of_collapsing_to_one(self, tmp_path, monkeypatch, capsys):
        # Simulates a previous run whose benchmark pool had already grown
        # past 1 entry (e.g. several improved checks in a row) by saving a
        # 3-member pool directly, then resumes via the real CLI (main())
        # and confirms the resumed run's own re-saved pool still has all 3
        # members -- not silently reset to a single-entry pool the way it
        # would be without this feature.
        out_dir = str(tmp_path / "cfr_runs")
        monkeypatch.setattr("sys.argv", [
            "cfr_main.py", "--iterations", "1", "--out-dir", out_dir,
            "--no-reload-previous", *_TINY_TRAINING_ARGS,
        ])
        main()
        capsys.readouterr()

        checkpoint_path = os.path.join(out_dir, "checkpoint_latest")
        config = self._config()
        rng = np.random.default_rng(1)
        nets = [Trainer.new(config, rng).net for _ in range(3)]
        net_config = cfr_networks.AdvantageNetConfig(
            feature_keys=config.feature_keys, hidden_sizes=config.hidden_sizes, table_size=config.table_size,
        )
        _save_benchmark_pool(nets, net_config, checkpoint_path)

        monkeypatch.setattr("sys.argv", [
            "cfr_main.py", "--iterations", "1", "--out-dir", out_dir,
            # enabled (nonzero), but --iterations 1 means it never actually
            # triggers a check this run -- only the reload is under test.
            *_tiny_args_with(benchmark_interval=100),
        ])
        main()

        assert len(_load_benchmark_pool(checkpoint_path)) == 3
