import os

import numpy as np
import pytest
import torch

import cfr_networks
from cfr_main import (
    _benchmark_pool_weights,
    _load_benchmark_pool,
    _reload_checkpoint,
    _resolve_benchmark_pool,
    _retire_stale_pool_members,
    _save_benchmark_pool,
    main,
    parse_args,
)
from cfr_reservoir import UNKNOWN_ITERATION, ReservoirBuffer
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
    benchmark pool (see cfr_main.py's own module docstring) -- each member
    now paired with its own start_iteration -- so a resumed run picks its
    pool up exactly where a previous run of the same checkpoint left it,
    rather than silently collapsing back to a single entry -- "a resumed
    run must not differ from one that simply kept running in the same
    process" applies to the pool just as much as it does to the
    net/reservoir/optimizer state _reload_checkpoint already covers
    above."""

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

        _save_benchmark_pool([(0, net_a), (5, net_b)], net_config, checkpoint_path)
        reloaded = _load_benchmark_pool(checkpoint_path)

        assert len(reloaded) == 2
        reloaded_starts = [start for start, _net in reloaded]
        assert reloaded_starts == [0, 5]
        assert _nets_have_equal_weights(reloaded[0][1], net_a)  # oldest first
        assert _nets_have_equal_weights(reloaded[1][1], net_b)

    def test_no_saved_pool_returns_empty_list(self, tmp_path):
        assert _load_benchmark_pool(str(tmp_path / "checkpoint_latest")) == []

    def test_shrinking_the_pool_removes_stale_members_from_disk(self, tmp_path):
        # A later save with fewer members (e.g. stale members were retired
        # between runs) must not leave the old, now-orphaned member files
        # behind -- _load_benchmark_pool would otherwise resurrect them on
        # the next resume even though they're no longer part of the pool.
        config = self._config()
        rng = np.random.default_rng(0)
        nets = [(i, Trainer.new(config, rng).net) for i in range(3)]
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
        nets = [(i, Trainer.new(config, rng).net) for i in range(3)]
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


def _make_reservoir(rows: list[tuple[float, float]]) -> ReservoirBuffer:
    """A ReservoirBuffer holding exactly `rows` (t, iteration) pairs, in
    order -- capacity == len(rows) so Algorithm R never replaces anything,
    keeping insertion order deterministic for these tests."""
    buf = ReservoirBuffer(capacity=max(len(rows), 1), feature_dim=1, num_actions=1, rng=np.random.default_rng(0))
    for t, iteration in rows:
        buf.add(np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32), np.array([True]), t, iteration=iteration)
    return buf


class TestBenchmarkPoolWeights:
    """_benchmark_pool_weights attributes each currently-held reservoir row
    to whichever pool member's own [start_i, start_{i+1}) span its raw
    iteration falls in, weighted by cfr_train._train_step's own Linear-CFR
    normalization (t / current_iteration) -- see cfr_main.py's own
    docstring and _benchmark_pool_weights' own docstring."""

    def test_empty_pool_returns_empty_array(self):
        reservoir = _make_reservoir([(1.0, 0.0)])
        assert list(_benchmark_pool_weights([], reservoir, current_iteration=1)) == []

    def test_empty_reservoir_returns_all_zero(self):
        pool = [(0, "net_a"), (5, "net_b")]
        reservoir = _make_reservoir([])
        assert list(_benchmark_pool_weights(pool, reservoir, current_iteration=10)) == [0.0, 0.0]

    def test_attributes_rows_to_the_member_whose_span_they_fall_in(self):
        pool = [(0, "net_a"), (10, "net_b"), (20, "net_c")]
        reservoir = _make_reservoir([
            (5.0, 3.0),  # net_a's span: [0, 10)
            (8.0, 7.0),  # net_a's span
            (12.0, 15.0),  # net_b's span: [10, 20)
            (25.0, 25.0),  # net_c's span: [20, inf)
        ])
        weights = _benchmark_pool_weights(pool, reservoir, current_iteration=25)
        assert weights[0] == pytest.approx((5.0 + 8.0) / 25.0)
        assert weights[1] == pytest.approx(12.0 / 25.0)
        assert weights[2] == pytest.approx(25.0 / 25.0)

    def test_rows_with_unknown_iteration_are_excluded(self):
        pool = [(0, "net_a")]
        reservoir = _make_reservoir([(5.0, 3.0)])
        reservoir.iterations[0] = UNKNOWN_ITERATION  # simulate a pre-upgrade row
        weights = _benchmark_pool_weights(pool, reservoir, current_iteration=10)
        assert weights[0] == 0.0


class TestRetireStalePoolMembers:
    def test_single_member_pool_is_never_retired(self):
        pool = [(0, "net_a")]
        assert _retire_stale_pool_members(pool, np.array([0.0]), min_weight_fraction=0.5) == pool

    def test_zero_total_weight_keeps_everything(self):
        pool = [(0, "net_a"), (10, "net_b")]
        assert _retire_stale_pool_members(pool, np.array([0.0, 0.0]), min_weight_fraction=0.5) == pool

    def test_drops_members_below_the_threshold_share(self):
        pool = [(0, "net_a"), (10, "net_b"), (20, "net_c")]
        weights = np.array([0.005, 0.5, 0.495])  # net_a's own share is ~0.5%, below 1%
        retained = _retire_stale_pool_members(pool, weights, min_weight_fraction=0.01)
        assert retained == [(10, "net_b"), (20, "net_c")]

    def test_most_recent_member_is_always_kept_even_with_zero_weight(self):
        pool = [(0, "net_a"), (10, "net_b"), (20, "net_c")]
        weights = np.array([1.0, 0.0, 0.0])  # net_c, the most recent, has no attributable weight yet
        retained = _retire_stale_pool_members(pool, weights, min_weight_fraction=0.01)
        assert retained == [(0, "net_a"), (20, "net_c")]


class TestResolveBenchmarkPool:
    def test_retains_and_normalizes_weights_proportionally(self):
        pool = [(0, "net_a"), (10, "net_b")]
        reservoir = _make_reservoir([(5.0, 3.0), (15.0, 12.0)])
        retained, weights = _resolve_benchmark_pool(pool, reservoir, current_iteration=15, min_weight_fraction=0.01)
        assert retained == pool
        assert list(weights) == pytest.approx([5.0 / 20.0, 15.0 / 20.0])
        assert weights.sum() == pytest.approx(1.0)

    def test_falls_back_to_uniform_when_reservoir_has_no_attributable_rows(self):
        pool = [(0, "net_a"), (10, "net_b"), (20, "net_c")]
        reservoir = _make_reservoir([])
        retained, weights = _resolve_benchmark_pool(pool, reservoir, current_iteration=25, min_weight_fraction=0.01)
        assert retained == pool
        assert list(weights) == pytest.approx([1 / 3, 1 / 3, 1 / 3])

    def test_retirement_shrinks_pool_and_reweights_the_survivors(self):
        pool = [(0, "net_a"), (10, "net_b"), (20, "net_c")]
        reservoir = _make_reservoir([
            (0.1, 3.0),  # a tiny share attributed to net_a once current_iteration is large
            (15.0, 15.0),  # net_b
            (25.0, 25.0),  # net_c
        ])
        retained, weights = _resolve_benchmark_pool(pool, reservoir, current_iteration=1000, min_weight_fraction=0.01)
        assert retained == [(10, "net_b"), (20, "net_c")]
        assert list(weights) == pytest.approx([15.0 / 40.0, 25.0 / 40.0])
