import os

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

import cfr_features
import cfr_networks
import cfr_reservoir
import strategy

_APP_PATH = os.path.join(os.path.dirname(__file__), "..", "poker_ga", "cfr_explorer.py")
_FEATURE_KEYS = ("street_norm", "facing_bet", "hand_category_norm")


def _make_checkpoint(path: str, rng: np.random.Generator, num_samples: int = 200) -> None:
    feature_dim = len(cfr_features.feature_indices(_FEATURE_KEYS))
    net = cfr_networks.AdvantageNet(input_dim=feature_dim, hidden_sizes=(8, 8))
    net_config = cfr_networks.AdvantageNetConfig(feature_keys=_FEATURE_KEYS, hidden_sizes=(8, 8), table_size=3)
    cfr_networks.save(net, net_config, path)

    reservoir = cfr_reservoir.ReservoirBuffer(
        capacity=num_samples, feature_dim=feature_dim, num_actions=strategy.NUM_ACTION_CATEGORIES, rng=rng,
    )
    for _ in range(num_samples):
        features = rng.random(feature_dim).astype(np.float32)
        regrets = rng.normal(size=strategy.NUM_ACTION_CATEGORIES).astype(np.float32)
        legal_mask = rng.random(strategy.NUM_ACTION_CATEGORIES) > 0.3
        legal_mask[strategy.ACTION_CALL] = True  # always legal, matching cfr_actions.legal_action_categories
        reservoir.add(features, regrets, legal_mask, float(rng.integers(1, 10)))
    reservoir.save(path)


@pytest.fixture(scope="module")
def _synthetic_checkpoint_path(tmp_path_factory) -> str:
    # Module-scoped and shared across every test that uses it (unlike a
    # fresh tmp_path per test) so Streamlit's cache_resource -- keyed on
    # this exact path string -- actually hits after the first test, instead
    # of recomputing SHAP feature importance (the slow part) from scratch
    # for all 200+ samples on every single test.
    path = os.path.join(str(tmp_path_factory.mktemp("cfr_explorer_checkpoint")), "checkpoint")
    _make_checkpoint(path, np.random.default_rng(0))
    return path


@pytest.fixture
def synthetic_checkpoint(_synthetic_checkpoint_path, monkeypatch):
    monkeypatch.setenv("CFR_EXPLORER_CHECKPOINT_PATH", _synthetic_checkpoint_path)
    return _synthetic_checkpoint_path


@pytest.fixture(scope="module")
def _empty_reservoir_checkpoint_path(tmp_path_factory) -> str:
    path = os.path.join(str(tmp_path_factory.mktemp("cfr_explorer_empty_checkpoint")), "checkpoint")
    _make_checkpoint(path, np.random.default_rng(0), num_samples=0)
    return path


@pytest.fixture
def empty_reservoir_checkpoint(_empty_reservoir_checkpoint_path, monkeypatch):
    monkeypatch.setenv("CFR_EXPLORER_CHECKPOINT_PATH", _empty_reservoir_checkpoint_path)
    return _empty_reservoir_checkpoint_path


def _run_app() -> AppTest:
    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=60)
    return at


class TestAppLoadsWithoutError:
    def test_no_exceptions_on_initial_run(self, synthetic_checkpoint):
        at = _run_app()
        assert not at.exception

    def test_reports_loaded_sample_count(self, synthetic_checkpoint):
        at = _run_app()
        assert any("200 reservoir samples loaded" in c.value for c in at.caption)

    def test_one_role_selectbox_per_configured_feature(self, synthetic_checkpoint):
        at = _run_app()
        role_boxes = [sb for sb in at.sidebar.selectbox if sb.key and sb.key.startswith("role::")]
        assert {sb.key for sb in role_boxes} == {f"role::{k}" for k in _FEATURE_KEYS}

    def test_empty_reservoir_shows_warning_not_a_crash(self, empty_reservoir_checkpoint):
        at = _run_app()
        assert not at.exception
        assert len(at.warning) >= 1

    def test_missing_checkpoint_shows_error_not_a_crash(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CFR_EXPLORER_CHECKPOINT_PATH", os.path.join(str(tmp_path), "does_not_exist"))
        at = _run_app()
        assert not at.exception
        assert len(at.error) >= 1


class TestInteraction:
    def test_marking_a_feature_as_group_split_still_renders(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::street_norm").set_value("Group split")
        at.run(timeout=60)
        assert not at.exception
        assert len(at.subheader) > 0  # one subheader per observed street value

    def test_marking_a_feature_as_table_split_renders_a_dataframe(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::hand_category_norm").set_value("Table split")
        at.run(timeout=60)
        assert not at.exception
        assert len(at.dataframe) > 0

    def test_two_table_splits_render_a_pivot_without_error(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::street_norm").set_value("Table split")
        at.sidebar.selectbox(key="role::hand_category_norm").set_value("Table split")
        at.run(timeout=60)
        assert not at.exception

    def test_a_third_table_split_shows_an_error_but_does_not_crash(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::street_norm").set_value("Table split")
        at.sidebar.selectbox(key="role::facing_bet").set_value("Table split")
        at.sidebar.selectbox(key="role::hand_category_norm").set_value("Table split")
        at.run(timeout=60)
        assert not at.exception
        assert len(at.error) >= 1

    def test_deselecting_all_filter_values_shows_a_no_match_warning(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::facing_bet").set_value("Filter")
        at.run(timeout=60)
        at.sidebar.multiselect(key="filter::facing_bet").set_value([])
        at.run(timeout=60)
        assert not at.exception
        assert len(at.warning) >= 1

    def test_collapse_toggle_still_renders_without_error(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::street_norm").set_value("Table split")
        at.run(timeout=60)
        at.sidebar.toggle[0].set_value(True)
        at.run(timeout=60)
        assert not at.exception
