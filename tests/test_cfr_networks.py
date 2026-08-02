import os
import tempfile

import numpy as np
import torch

import strategy
from cfr_networks import AdvantageNet, AdvantageNetConfig, load, save


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
