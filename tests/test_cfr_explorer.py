import os
import re

import numpy as np
import pytest
import torch
from streamlit.testing.v1 import AppTest

import cfr_features
import cfr_networks
import cfr_reservoir
import strategy

_APP_PATH = os.path.join(os.path.dirname(__file__), "..", "poker_ga", "cfr_explorer.py")
_FEATURE_KEYS = ("street_norm", "facing_bet", "hand_category_norm")


def _shap_values_by_key(at: AppTest) -> dict[str, float]:
    role_boxes = [sb for sb in at.sidebar.selectbox if sb.key and sb.key.startswith("role::")]
    return {
        sb.key.removeprefix("role::"): float(re.search(r"SHAP ([\d.]+)\)", sb.label).group(1))
        for sb in role_boxes
    }


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


def _make_correlated_checkpoint(path: str, rng: np.random.Generator, num_samples: int = 400) -> None:
    """Like _make_checkpoint, except hand_category_norm is pinned to
    exactly 0.0 whenever street_norm's own value falls in the "Preflop"
    bucket -- mirroring a real feature like num_overcards_norm, which is
    always exactly 0 preflop since there's no board yet to have overcards
    on. Filtering to Preflop should then make hand_category_norm's SHAP
    contribution collapse to ~0."""
    feature_dim = len(cfr_features.feature_indices(_FEATURE_KEYS))
    torch.manual_seed(0)  # AdvantageNet's init otherwise draws from torch's unseeded global RNG
    net = cfr_networks.AdvantageNet(input_dim=feature_dim, hidden_sizes=(8, 8))
    street_idx = _FEATURE_KEYS.index("street_norm")
    hand_idx = _FEATURE_KEYS.index("hand_category_norm")
    with torch.no_grad():
        net.model[0].weight[:, hand_idx] *= 20.0  # give it an outsized true effect over the whole reservoir
    net_config = cfr_networks.AdvantageNetConfig(feature_keys=_FEATURE_KEYS, hidden_sizes=(8, 8), table_size=3)
    cfr_networks.save(net, net_config, path)
    reservoir = cfr_reservoir.ReservoirBuffer(
        capacity=num_samples, feature_dim=feature_dim, num_actions=strategy.NUM_ACTION_CATEGORIES, rng=rng,
    )
    for _ in range(num_samples):
        features = rng.random(feature_dim).astype(np.float32)
        if cfr_features.bucket_label("street_norm", features[street_idx]) == "Preflop":
            features[hand_idx] = 0.0
        regrets = rng.normal(size=strategy.NUM_ACTION_CATEGORIES).astype(np.float32)
        legal_mask = rng.random(strategy.NUM_ACTION_CATEGORIES) > 0.3
        legal_mask[strategy.ACTION_CALL] = True
        reservoir.add(features, regrets, legal_mask, float(rng.integers(1, 10)))
    reservoir.save(path)


@pytest.fixture(scope="module")
def _correlated_checkpoint_path(tmp_path_factory) -> str:
    path = os.path.join(str(tmp_path_factory.mktemp("cfr_explorer_correlated_checkpoint")), "checkpoint")
    _make_correlated_checkpoint(path, np.random.default_rng(0))
    return path


@pytest.fixture
def correlated_checkpoint(_correlated_checkpoint_path, monkeypatch):
    monkeypatch.setenv("CFR_EXPLORER_CHECKPOINT_PATH", _correlated_checkpoint_path)
    return _correlated_checkpoint_path


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


class TestFilteredFeatureImportance:
    def test_narrowing_a_filter_changes_reported_shap_values(self, synthetic_checkpoint):
        at = _run_app()
        before = _shap_values_by_key(at)

        at.sidebar.selectbox(key="role::street_norm").set_value("Filter")
        at.run(timeout=60)
        observed = at.sidebar.multiselect(key="filter::street_norm").value
        at.sidebar.multiselect(key="filter::street_norm").set_value(observed[:1])
        at.run(timeout=60)

        after = _shap_values_by_key(at)
        assert not at.exception
        assert before != after

    def test_zero_matching_rows_still_shows_every_role_selectbox(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::facing_bet").set_value("Filter")
        at.run(timeout=60)
        at.sidebar.multiselect(key="filter::facing_bet").set_value([])
        at.run(timeout=60)

        role_boxes = [sb for sb in at.sidebar.selectbox if sb.key and sb.key.startswith("role::")]
        assert {sb.key for sb in role_boxes} == {f"role::{k}" for k in _FEATURE_KEYS}
        shap_values = _shap_values_by_key(at)
        assert all(v == 0.0 for v in shap_values.values())

    def test_feature_pinned_constant_by_the_filter_scores_exactly_zero(self, correlated_checkpoint):
        # Regression test: hand_category_norm is pinned to a single constant
        # value whenever street_norm falls in the Preflop bucket (see
        # _make_correlated_checkpoint). When the SHAP background reference
        # is drawn from the same filtered pool as the explained rows (the
        # fix), x_i - background_i is exactly 0 for this feature at every
        # interpolation point, so its contribution is exactly 0.0 -- not
        # just small. Drawing background from the whole, unfiltered
        # reservoir instead (where hand_category_norm does vary) leaves a
        # nonzero residual here, since mean-centering alone doesn't cancel
        # the per-row noise that mismatch introduces.
        at = _run_app()
        unfiltered = _shap_values_by_key(at)

        at.sidebar.selectbox(key="role::street_norm").set_value("Filter")
        at.run(timeout=60)
        at.sidebar.multiselect(key="filter::street_norm").set_value(["Preflop"])
        at.run(timeout=60)

        assert not at.exception
        filtered = _shap_values_by_key(at)
        assert unfiltered["hand_category_norm"] > 0.1  # meaningfully important over the whole reservoir
        assert filtered["hand_category_norm"] == 0.0  # constant, and so exactly uninformative, once filtered to Preflop


class TestActiveFiltersWidget:
    def test_no_widget_when_no_filters_active(self, synthetic_checkpoint):
        at = _run_app()
        assert len(at.multiselect) == 0

    def test_active_filter_shows_a_removable_tag_widget(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::facing_bet").set_value("Filter")
        at.run(timeout=60)

        active_widgets = [ms for ms in at.multiselect if ms.key and ms.key.startswith("active_filters::")]
        assert len(active_widgets) == 1
        assert active_widgets[0].value == ["facing_bet"]

    def test_removing_a_tag_turns_that_filter_back_to_unused(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::facing_bet").set_value("Filter")
        at.run(timeout=60)
        active_widget = next(ms for ms in at.multiselect if ms.key and ms.key.startswith("active_filters::"))
        active_widget.set_value([])
        at.run(timeout=60)

        assert not at.exception
        assert at.sidebar.selectbox(key="role::facing_bet").value == "Unused"
        assert len(at.multiselect) == 0


class TestHierarchicalGroupSplit:
    def test_single_group_split_still_uses_a_subheader(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::street_norm").set_value("Group split")
        at.run(timeout=60)
        assert not at.exception
        assert len(at.subheader) > 0

    def test_two_group_splits_nest_a_markdown_heading_under_the_subheader(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::street_norm").set_value("Group split")
        at.sidebar.selectbox(key="role::hand_category_norm").set_value("Group split")
        at.run(timeout=60)

        assert not at.exception
        assert len(at.subheader) > 0
        assert len(at.markdown) > 0
        # The old flat behavior joined every key/value pair onto one combined
        # heading line (e.g. "Street = Flop, Hand Strength Tier = Pair") --
        # hierarchical nesting means no single heading repeats both keys.
        headings = [el.value for el in list(at.subheader) + list(at.markdown)]
        assert not any("Street" in h and "Hand Strength Tier" in h for h in headings)
