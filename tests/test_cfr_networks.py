import os
import tempfile

import numpy as np
import pytest
import torch

import strategy
from cfr_networks import (
    AdvantageNet, AdvantageNetConfig, _normalized_mean_abs_shap, clone, load, mean_shap_contributions, save,
)
from cfr_reservoir import ReservoirBuffer


class TestAdvantageNetForward:
    def test_output_shape_matches_num_action_categories(self):
        net = AdvantageNet(input_dim=5, hidden_sizes=(8, 8))
        x = torch.zeros((3, 5))
        out = net(x)
        assert out.shape == (3, strategy.NUM_ACTION_CATEGORIES)

    def test_predict_returns_1d_numpy_array(self):
        net = AdvantageNet(input_dim=4, hidden_sizes=(8,))
        features = np.zeros(4, dtype=np.float32)
        out = net.predict(features)
        assert isinstance(out, np.ndarray)
        assert out.shape == (strategy.NUM_ACTION_CATEGORIES,)

    def test_zero_hidden_layers_is_a_linear_model(self):
        net = AdvantageNet(input_dim=3, hidden_sizes=())
        x = torch.zeros((1, 3))
        out = net(x)
        assert out.shape == (1, strategy.NUM_ACTION_CATEGORIES)


class TestSaveLoadRoundTrip:
    def test_reloaded_net_produces_identical_predictions(self):
        net = AdvantageNet(input_dim=6, hidden_sizes=(16, 16))
        config = AdvantageNetConfig(
            feature_keys=("hand_category_norm", "street_norm", "facing_bet", "pot_type_norm", "call_amount_norm", "spr_norm"),
            hidden_sizes=(16, 16),
            table_size=6,
        )
        features = np.random.default_rng(0).random(6).astype(np.float32)
        original_prediction = net.predict(features)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "checkpoint")
            save(net, config, path)
            assert os.path.exists(f"{path}.pt")
            assert os.path.exists(f"{path}.json")

            loaded_net, loaded_config = load(path)
            reloaded_prediction = loaded_net.predict(features)

        assert loaded_config.feature_keys == config.feature_keys
        assert loaded_config.hidden_sizes == config.hidden_sizes
        assert loaded_config.table_size == config.table_size
        assert np.allclose(original_prediction, reloaded_prediction)

    def test_loaded_input_dim_matches_feature_key_count(self):
        keys = ("hand_category_norm", "street_norm")
        net = AdvantageNet(input_dim=len(keys), hidden_sizes=(8,))
        config = AdvantageNetConfig(feature_keys=keys, hidden_sizes=(8,), table_size=2)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "checkpoint")
            save(net, config, path)
            loaded_net, _ = load(path)
        assert loaded_net.input_dim == 2


class TestClone:
    def test_clone_produces_identical_predictions(self):
        net = AdvantageNet(input_dim=5, hidden_sizes=(8, 8))
        features = np.random.default_rng(0).random(5).astype(np.float32)
        cloned = clone(net)
        assert np.allclose(net.predict(features), cloned.predict(features))

    def test_clone_is_independent_of_later_training_on_the_original(self):
        net = AdvantageNet(input_dim=4, hidden_sizes=(8,))
        features = np.random.default_rng(0).random(4).astype(np.float32)
        cloned = clone(net)
        before = cloned.predict(features)

        optimizer = torch.optim.SGD(net.parameters(), lr=1.0)
        net.train()
        loss = net(torch.from_numpy(features).unsqueeze(0)).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        after = cloned.predict(features)
        assert np.allclose(before, after)  # unaffected by training on the original
        assert not np.allclose(net.predict(features), cloned.predict(features))  # original actually did change


def _filled_reservoir(capacity, feature_dim, rng):
    buf = ReservoirBuffer(capacity=capacity, feature_dim=feature_dim, num_actions=strategy.NUM_ACTION_CATEGORIES, rng=rng)
    for _ in range(capacity):
        buf.add(
            rng.random(feature_dim).astype(np.float32),
            rng.random(strategy.NUM_ACTION_CATEGORIES).astype(np.float32),
            rng.random(strategy.NUM_ACTION_CATEGORIES) > 0.5,
            float(rng.integers(1, 10)),
        )
    return buf


class TestNormalizedMeanAbsShap:
    def test_feature_constant_across_samples_scores_zero_despite_large_raw_value(self):
        # 2 actions, 5 samples, 3 features -- feature 1 contributes a
        # constant +20 to every sample/action, exactly the "looks important
        # but is actually uninformative for this pool" case.
        shap_values = [
            np.array([[1.0, 20.0, -3.0], [2.0, 20.0, 3.0], [-1.0, 20.0, 0.5], [0.5, 20.0, -2.0], [-2.0, 20.0, 1.0]]),
            np.array([[0.2, 20.0, 1.0], [-0.3, 20.0, -1.0], [0.1, 20.0, 2.0], [-0.2, 20.0, 0.0], [0.4, 20.0, -0.5]]),
        ]

        mean_abs = _normalized_mean_abs_shap(shap_values)

        assert mean_abs[1] == pytest.approx(0.0, abs=1e-9)
        assert mean_abs[0] > 0.0
        assert mean_abs[2] > 0.0

    def test_unnormalized_would_have_ranked_the_constant_feature_highest(self):
        # Same data as above: sanity-check that this scenario really would
        # fool a plain mean(|raw shap|) metric, motivating the normalization.
        shap_values = [
            np.array([[1.0, 20.0, -3.0], [2.0, 20.0, 3.0], [-1.0, 20.0, 0.5], [0.5, 20.0, -2.0], [-2.0, 20.0, 1.0]]),
            np.array([[0.2, 20.0, 1.0], [-0.3, 20.0, -1.0], [0.1, 20.0, 2.0], [-0.2, 20.0, 0.0], [0.4, 20.0, -0.5]]),
        ]
        raw_mean_abs = np.mean(np.abs(np.stack(shap_values, axis=0)), axis=(0, 1))
        assert np.argmax(raw_mean_abs) == 1

        mean_abs = _normalized_mean_abs_shap(shap_values)
        assert np.argmax(mean_abs) != 1

    def test_varying_feature_unaffected_when_its_own_mean_is_zero(self):
        shap_values = [np.array([[1.0], [-1.0], [2.0], [-2.0]])]
        mean_abs = _normalized_mean_abs_shap(shap_values)
        assert mean_abs[0] == pytest.approx(1.5)  # mean(|1|,|1|,|2|,|2|) unchanged by centering on a zero mean


class TestMeanShapContributions:
    def test_empty_reservoir_returns_empty_list(self):
        rng = np.random.default_rng(0)
        net = AdvantageNet(input_dim=4, hidden_sizes=(8,))
        buf = ReservoirBuffer(capacity=10, feature_dim=4, num_actions=strategy.NUM_ACTION_CATEGORIES, rng=rng)
        keys = ("a", "b", "c", "d")
        assert mean_shap_contributions(net, buf, keys, rng, sample_size=5, background_size=2, nsamples=5) == []

    def test_returns_one_entry_per_feature_sorted_descending_and_nonnegative(self):
        rng = np.random.default_rng(0)
        net = AdvantageNet(input_dim=4, hidden_sizes=(8,))
        buf = _filled_reservoir(capacity=30, feature_dim=4, rng=rng)
        keys = ("a", "b", "c", "d")

        result = mean_shap_contributions(net, buf, keys, rng, sample_size=10, background_size=5, nsamples=5)

        assert sorted(k for k, _ in result) == sorted(keys)
        values = [v for _, v in result]
        assert all(v >= 0.0 for v in values)
        assert values == sorted(values, reverse=True)

    def test_a_feature_the_net_structurally_ignores_ranks_last_near_zero(self):
        rng = np.random.default_rng(1)
        net = AdvantageNet(input_dim=4, hidden_sizes=(8,))
        with torch.no_grad():
            first_layer = net.model[0]
            first_layer.weight[:, 2] = 0.0  # feature index 2 can never affect any hidden unit
        buf = _filled_reservoir(capacity=30, feature_dim=4, rng=rng)
        keys = ("a", "b", "ignored", "d")

        result = mean_shap_contributions(net, buf, keys, rng, sample_size=10, background_size=5, nsamples=5)

        by_key = dict(result)
        assert by_key["ignored"] == pytest.approx(0.0, abs=1e-6)
        assert result[-1][0] == "ignored"
