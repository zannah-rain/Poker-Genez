import os

import numpy as np
import pytest

from cfr_main import _reload_checkpoint, main, parse_args
from cfr_train import DeepCFRConfig, Trainer
from game import GameConfig

_FEATURE_KEYS = "hand_category_norm,street_norm,facing_bet,pot_type_norm"

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
