import numpy as np
import torch

from cfr_reservoir import ReservoirBuffer


def _make_buffer(capacity, rng_seed=0):
    return ReservoirBuffer(capacity=capacity, feature_dim=3, num_actions=2, rng=np.random.default_rng(rng_seed))


def _add(buf, value, t=1.0):
    features = np.full(3, value, dtype=np.float32)
    regrets = np.full(2, value, dtype=np.float32)
    legal_mask = np.array([True, value >= 0])
    buf.add(features, regrets, legal_mask, t)


class TestCapacity:
    def test_fills_up_to_capacity_before_replacing(self):
        buf = _make_buffer(capacity=5)
        for i in range(5):
            _add(buf, i)
        assert len(buf) == 5

    def test_never_exceeds_capacity(self):
        buf = _make_buffer(capacity=5)
        for i in range(50):
            _add(buf, i)
        assert len(buf) == 5

    def test_below_capacity_keeps_every_item_in_insertion_order(self):
        buf = _make_buffer(capacity=10)
        for i in range(4):
            _add(buf, i)
        assert list(buf.features[:4, 0]) == [0.0, 1.0, 2.0, 3.0]


class TestUniformReplacement:
    def test_every_slot_eventually_gets_replaced(self):
        buf = _make_buffer(capacity=8, rng_seed=1)
        for i in range(8):
            _add(buf, -1)  # fill with sentinel value -1
        for i in range(20_000):
            _add(buf, i)
        # After many more inserts than capacity, none of the original
        # sentinel-filled slots should statistically survive.
        assert not np.any(buf.features[:, 0] == -1.0)

    def test_stores_the_provided_weight(self):
        buf = _make_buffer(capacity=4)
        _add(buf, 1, t=7.0)
        assert buf.weights[0] == 7.0

    def test_stores_legal_mask(self):
        buf = _make_buffer(capacity=4)
        _add(buf, -1)  # legal_mask = [True, False] since value < 0
        assert list(buf.legal_masks[0]) == [True, False]


class TestSample:
    def test_returns_torch_tensors_of_requested_batch_size(self):
        buf = _make_buffer(capacity=10)
        for i in range(10):
            _add(buf, i)
        features, regrets, legal_mask, weights = buf.sample(4)
        assert isinstance(features, torch.Tensor)
        assert features.shape == (4, 3)
        assert regrets.shape == (4, 2)
        assert legal_mask.shape == (4, 2)
        assert weights.shape == (4,)

    def test_batch_size_larger_than_buffer_is_clipped(self):
        buf = _make_buffer(capacity=10)
        for i in range(3):
            _add(buf, i)
        features, regrets, legal_mask, weights = buf.sample(100)
        assert features.shape[0] == 3

    def test_sampled_features_come_from_stored_data(self):
        buf = _make_buffer(capacity=10)
        for i in range(10):
            _add(buf, i)
        features, *_ = buf.sample(5)
        stored_values = set(buf.features[: len(buf), 0].tolist())
        for row in features:
            assert float(row[0].item()) in stored_values
