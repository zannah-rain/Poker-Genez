import os
import tempfile

import numpy as np
import torch

import cfr_networks
import cfr_reservoir
from cfr_train import DeepCFRConfig, Trainer, run_iteration
from game import GameConfig

_FEATURE_KEYS = ("hand_category_norm", "street_norm", "call_amount_norm", "total_raises_norm")


def _tiny_config(**overrides) -> DeepCFRConfig:
    kwargs = dict(
        feature_keys=_FEATURE_KEYS,
        hidden_sizes=(8,),
        table_size=2,
        iterations=2,
        traversals_per_iteration=2,
        sgd_steps_per_iteration=1,
        batch_size=8,
        reservoir_capacity=50,
        num_equity_rollouts=1,
        game_config=GameConfig(),
    )
    kwargs.update(overrides)
    return DeepCFRConfig(**kwargs)


class TestTrainerSaveLoadRoundTrip:
    def test_fresh_trainer_starts_at_zero_completed_iterations(self):
        trainer = Trainer.new(_tiny_config(), np.random.default_rng(0))
        assert trainer.completed_iterations == 0

    def test_save_writes_a_trainer_state_file(self):
        trainer = Trainer.new(_tiny_config(), np.random.default_rng(0))
        trainer.completed_iterations = 7
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "checkpoint")
            trainer.save(path)
            assert os.path.exists(f"{path}_trainer_state.pt")

    def test_load_trainer_state_round_trips_completed_iterations_and_optimizer(self):
        trainer = Trainer.new(_tiny_config(), np.random.default_rng(0))
        # Take a real gradient step so the optimizer actually has state
        # (Adam's momentum/variance buffers) worth checking survives the
        # round trip, not just an empty state_dict.
        rng = np.random.default_rng(0)
        run_iteration(trainer, rng, 1, show_progress=False)
        trainer.completed_iterations = 42
        trainer.generation = 3

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "checkpoint")
            trainer.save(path)
            optimizer_state, completed_iterations, generation = Trainer.load_trainer_state(path)

        assert completed_iterations == 42
        assert generation == 3
        assert optimizer_state is not None
        # The reloaded state dict should reconstruct an optimizer with the
        # same momentum/variance buffers as the one that was saved.
        fresh_net = cfr_networks.AdvantageNet(input_dim=len(_FEATURE_KEYS), hidden_sizes=(8,))
        fresh_net.load_state_dict(trainer.net.state_dict())
        reloaded_optimizer = torch.optim.Adam(fresh_net.parameters(), lr=1e-3)
        reloaded_optimizer.load_state_dict(optimizer_state)
        for group_before, group_after in zip(trainer.optimizer.state_dict()["state"].values(), optimizer_state["state"].values()):
            assert torch.allclose(group_before["exp_avg"], group_after["exp_avg"])

    def test_load_trainer_state_missing_file_returns_none_and_zero(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "no_such_checkpoint")
            optimizer_state, completed_iterations, generation = Trainer.load_trainer_state(path)
        assert optimizer_state is None
        assert completed_iterations == 0
        assert generation == 0

    def test_load_trainer_state_missing_generation_key_falls_back_to_zero(self):
        # Simulates a trainer-state file saved before `generation` existed
        # (but after completed_iterations already did) -- must fall back to
        # 0, the same safe bootstrap a fresh run starts from, not error out.
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "checkpoint")
            torch.save({"optimizer": {}, "completed_iterations": 9}, f"{path}_trainer_state.pt")
            optimizer_state, completed_iterations, generation = Trainer.load_trainer_state(path)
        assert completed_iterations == 9
        assert generation == 0

    def test_new_with_initial_optimizer_state_resumes_momentum(self):
        trainer = Trainer.new(_tiny_config(), np.random.default_rng(0))
        run_iteration(trainer, np.random.default_rng(0), 1, show_progress=False)
        saved_state = trainer.optimizer.state_dict()

        resumed = Trainer.new(
            _tiny_config(), np.random.default_rng(0),
            initial_net=trainer.net, initial_optimizer_state=saved_state, completed_iterations=1,
            initial_generation=2,
        )
        assert resumed.completed_iterations == 1
        assert resumed.generation == 2
        resumed_state = resumed.optimizer.state_dict()
        for group_before, group_after in zip(saved_state["state"].values(), resumed_state["state"].values()):
            assert torch.allclose(group_before["exp_avg"], group_after["exp_avg"])


class TestRunIterationTracksCompletedIterations:
    def test_run_iteration_sets_completed_iterations_to_the_passed_iteration(self):
        trainer = Trainer.new(_tiny_config(), np.random.default_rng(0))
        run_iteration(trainer, np.random.default_rng(0), 5, show_progress=False)
        assert trainer.completed_iterations == 5

    def test_resuming_from_a_high_iteration_does_not_inflate_reservoir_sample_weights(self):
        """Regression test for the reload loss-spike bug: a reservoir
        sample's stored weight is the outer iteration `t` it was collected
        at, times its own path_weight -- see cfr_reservoir.ReservoirBuffer.add
        and cfr_tree._decision_node's own docstring (path_weight is never
        more than 1.0, so this is an upper bound, not an exact figure --
        only a hand's very first traverser decision keeps the full `t`;
        every deeper one is discounted, and a small reservoir like this
        test's own can easily end up not holding any of the (rare, one per
        hand) undiscounted ones by chance) -- and _train_step normalizes by
        the *current* outer iteration. If a resumed run's iteration counter
        restarted at 1 instead of continuing from wherever the reservoir's
        own samples were collected, an old sample's normalized weight
        (t / current_iteration) would blow up far past 1.0 instead of
        staying in the (0, 1] range Linear CFR's weighting assumes."""
        trainer = Trainer.new(_tiny_config(), np.random.default_rng(0))
        # Simulate a reservoir carried over from a much longer previous run.
        rng = np.random.default_rng(0)
        run_iteration(trainer, rng, 300, show_progress=False)
        assert trainer.reservoir.weights[: trainer.reservoir.size].max() <= 300

        # A resumed run must keep counting from where the reservoir's own
        # samples left off, not restart at 1.
        next_iteration = trainer.completed_iterations + 1
        assert next_iteration == 301
        max_normalized_weight = trainer.reservoir.weights[: trainer.reservoir.size].max() / next_iteration
        assert max_normalized_weight <= 1.0


class TestRunIterationForwardsGeneration:
    def test_reservoir_samples_are_stamped_with_trainers_current_generation(self):
        trainer = Trainer.new(_tiny_config(), np.random.default_rng(0))
        trainer.generation = 3
        run_iteration(trainer, np.random.default_rng(0), 1, show_progress=False)
        assert trainer.reservoir.size > 0
        assert np.all(trainer.reservoir.generations[: trainer.reservoir.size] == 3)

    def test_run_iteration_never_advances_generation_itself(self):
        # Only cfr_main.py's own training loop advances generation, on a
        # confirmed-improved benchmark check -- run_iteration just forwards
        # whatever value is already there.
        trainer = Trainer.new(_tiny_config(), np.random.default_rng(0))
        run_iteration(trainer, np.random.default_rng(0), 1, show_progress=False)
        assert trainer.generation == 0
