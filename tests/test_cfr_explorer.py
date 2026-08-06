import json
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
_FEATURE_KEYS = ("street_norm", "hole_suited", "hand_category_norm")
# Mirrors _COLLAPSED_LABELS -- not imported directly since
# cfr_explorer.py calls main() at module scope (it's a script, driven only
# via AppTest.from_file, not a plain importable module).
_COLLAPSED_LABELS = ("Fold", "Call", "Raise", "All-In")


def _shap_values_by_key(at: AppTest) -> dict[str, float]:
    role_boxes = [sb for sb in at.sidebar.selectbox if sb.key and sb.key.startswith("role::")]
    return {
        sb.key.removeprefix("role::"): float(re.search(r"SHAP ([\d.]+)\)", sb.label).group(1))
        for sb in role_boxes
    }


def _strip_shap_suffix(option: str) -> str:
    """A "Add graph/table/filter" or "Split By" dropdown option's plain
    feature label, with its "  (SHAP 0.0000)" suffix (see
    _render_substrategy) removed -- so tests can check which features are
    offered without hardcoding an exact importance value."""
    return re.sub(r"\s+\(SHAP [\d.]+\)$", "", option)


def _batch_set(at: AppTest, values: dict[str, list[str]]) -> AppTest:
    """Sets several multiselect widgets' values together before a single
    .run() -- AppTest's own widget-state round-trip for a format_func-based
    multiselect (every widget in this app that shows a "(SHAP 0.0000)"
    suffix) isn't reliably preserved for a widget that's *not* re-asserted
    in the same batch/run as some other widget's change (confirmed by
    inspecting raw st.session_state directly during development -- a
    testing-harness quirk, not real app behavior; a real browser session
    doesn't have this problem, since it tracks each option's raw identity
    client-side instead of round-tripping through format_func on every
    interaction). The safe pattern used throughout this file: batch every
    widget that's being given its *first* value together in one .run(), and
    keep any dependent widget that only exists *after* an earlier one has a
    value (e.g. "keep values" only appearing once a claim feature is
    picked) to its own preceding .run()."""
    for key, value in values.items():
        at.multiselect(key=key).set_value(value)
    at.run(timeout=60)
    return at


def _substrategy_child_prefix(at: AppTest, parent_prefix: str, index: int = -1) -> str:
    """The key_prefix of the `index`-th (default: most recently added)
    child sub-strategy currently under `parent_prefix`."""
    child_id = at.session_state["substrategy_children"][parent_prefix][index]
    return f"{parent_prefix}substrategy_{child_id}::"


def _add_substrategy(at: AppTest, parent_prefix: str) -> str:
    """Clicks "Add sub-strategy" under `parent_prefix` and returns the new
    child's own key_prefix."""
    at.button(key=f"{parent_prefix}add_substrategy").click().run(timeout=60)
    return _substrategy_child_prefix(at, parent_prefix)


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


class TestSidebarRoles:
    def test_marking_a_feature_as_split_by_renders_a_dataframe(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::hand_category_norm").set_value("Split By")
        at.run(timeout=60)
        assert not at.exception
        assert len(at.dataframe) > 0

    def test_two_split_by_features_render_a_pivot_without_error(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::street_norm").set_value("Split By")
        at.sidebar.selectbox(key="role::hand_category_norm").set_value("Split By")
        at.run(timeout=60)
        assert not at.exception

    def test_a_third_split_by_shows_an_error_but_does_not_crash(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::street_norm").set_value("Split By")
        at.sidebar.selectbox(key="role::hole_suited").set_value("Split By")
        at.sidebar.selectbox(key="role::hand_category_norm").set_value("Split By")
        at.run(timeout=60)
        assert not at.exception
        assert any("Only 2 features can be used as Split By" in e.value for e in at.error)

    def test_deselecting_all_filter_values_shows_a_no_match_warning(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::hole_suited").set_value("Filter")
        at.run(timeout=60)
        at.sidebar.multiselect(key="filter::hole_suited").set_value([])
        at.run(timeout=60)
        assert not at.exception
        assert len(at.warning) >= 1

    def test_collapse_toggle_still_renders_without_error(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::street_norm").set_value("Split By")
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
        at.sidebar.selectbox(key="role::hole_suited").set_value("Filter")
        at.run(timeout=60)
        at.sidebar.multiselect(key="filter::hole_suited").set_value([])
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
        # meaningfully important over the whole reservoir -- AdvantageNet's
        # LayerNorm caps how dominant _make_correlated_checkpoint's single
        # amplified weight column can make one feature look (it normalizes
        # away the raw magnitude, not just the sign), so this precondition
        # is calibrated to that architecture's actual ceiling rather than
        # the much larger headroom a plain (unnormalized) MLP would allow.
        assert unfiltered["hand_category_norm"] > 0.01
        assert filtered["hand_category_norm"] == 0.0  # constant, and so exactly uninformative, once filtered to Preflop


class TestActiveFiltersWidget:
    def test_no_widget_when_no_filters_active(self, synthetic_checkpoint):
        at = _run_app()
        active_widgets = [ms for ms in at.multiselect if ms.key and ms.key.startswith("active_filters::")]
        assert len(active_widgets) == 0

    def test_active_filter_shows_a_removable_tag_widget(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::hole_suited").set_value("Filter")
        at.run(timeout=60)

        active_widgets = [ms for ms in at.multiselect if ms.key and ms.key.startswith("active_filters::")]
        assert len(active_widgets) == 1
        assert active_widgets[0].value == ["hole_suited"]

    def test_removing_a_tag_turns_that_filter_back_to_unused(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::hole_suited").set_value("Filter")
        at.run(timeout=60)
        active_widget = next(ms for ms in at.multiselect if ms.key and ms.key.startswith("active_filters::"))
        active_widget.set_value([])
        at.run(timeout=60)

        assert not at.exception
        assert at.sidebar.selectbox(key="role::hole_suited").value == "Unused"
        active_widgets = [ms for ms in at.multiselect if ms.key and ms.key.startswith("active_filters::")]
        assert len(active_widgets) == 0


class TestGraphs:
    def test_marking_a_feature_as_graph_renders_one_line_chart(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::street_norm").set_value("Graph")
        at.run(timeout=60)

        assert not at.exception
        charts = at.get("plotly_chart")
        assert len(charts) == 1
        assert charts[0].id.endswith("graph_chart::street_norm")

    def test_no_graphs_section_when_nothing_is_marked_graph(self, synthetic_checkpoint):
        at = _run_app()
        assert not at.get("plotly_chart")
        assert not any("Graphs" == h.value for h in at.header)

    def test_crossing_with_another_feature_adds_four_heatmaps(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::street_norm").set_value("Graph")
        at.run(timeout=60)
        at.multiselect(key="root::graph_cross::street_norm").set_value(["hand_category_norm"])
        at.run(timeout=60)

        assert not at.exception
        charts = at.get("plotly_chart")
        assert len(charts) == 5  # 1 line chart + 4 simplified-action heatmaps
        heat_ids = [c.id for c in charts if "graph_heat::street_norm::hand_category_norm::" in c.id]
        assert len(heat_ids) == 4

    def test_cross_dropdown_excludes_the_graphed_feature_itself(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::street_norm").set_value("Graph")
        at.run(timeout=60)

        cross = at.multiselect(key="root::graph_cross::street_norm")
        assert "Betting Street" not in cross.options  # street_norm's own label -- can't cross a feature with itself
        assert set(cross.options) == {"Suited Hole Cards", "Hand Strength Tier"}

    def test_line_chart_has_one_trace_per_action_and_a_percent_y_axis(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::street_norm").set_value("Graph")
        at.run(timeout=60)

        spec = json.loads(at.get("plotly_chart")[0].proto.spec)
        assert len(spec["data"]) == strategy.NUM_ACTION_CATEGORIES
        assert spec["layout"]["yaxis"]["range"] == [0, 1]
        assert spec["layout"]["yaxis"]["tickformat"] == ".0%"

    def test_collapsing_actions_also_collapses_the_line_chart_to_four_traces(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::street_norm").set_value("Graph")
        at.run(timeout=60)
        at.sidebar.toggle[0].set_value(True)
        at.run(timeout=60)

        spec = json.loads(at.get("plotly_chart")[0].proto.spec)
        assert len(spec["data"]) == 4
        assert [trace["name"] for trace in spec["data"]] == list(_COLLAPSED_LABELS)

    def test_heatmaps_always_use_the_four_simplified_actions_regardless_of_toggle(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::street_norm").set_value("Graph")
        at.run(timeout=60)
        at.multiselect(key="root::graph_cross::street_norm").set_value(["hand_category_norm"])
        at.run(timeout=60)
        at.sidebar.toggle[0].set_value(True)  # collapse toggle should only affect the line chart, not the heatmaps
        at.run(timeout=60)

        assert not at.exception
        heatmap_titles = sorted(
            json.loads(c.proto.spec)["layout"]["title"]["text"]
            for c in at.get("plotly_chart")
            if "graph_heat::" in c.id
        )
        assert heatmap_titles == sorted(f"{label} rate" for label in _COLLAPSED_LABELS)

    def test_root_graphs_section_uses_the_prominent_header(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::hole_suited").set_value("Graph")
        at.run(timeout=60)

        assert not at.exception
        assert any(h.value == "Graphs" for h in at.header)
        assert not any("Graphs" in m.value for m in at.markdown)

    def test_substrategy_graphs_heading_is_one_level_below_its_own_heading(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::hole_suited").set_value("Graph")
        at.run(timeout=60)
        _add_substrategy(at, "root::")

        assert not at.exception
        # The sub-strategy's own heading is a subheader (level 0); "Graphs"
        # for that same node should render one level smaller, as a markdown
        # heading, not compete with it as another st.header/subheader.
        assert not any(h.value == "Graphs" for h in at.subheader)
        graphs_headings = [m.value for m in at.markdown if "Graphs" in m.value]
        assert len(graphs_headings) == 1
        assert graphs_headings[0] == "#### Graphs"  # one level below level-0's implicit h3

    def test_each_substrategy_gets_its_own_independently_keyed_graph(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::hole_suited").set_value("Graph")
        at.run(timeout=60)
        c0 = _add_substrategy(at, "root::")
        c1 = _add_substrategy(at, "root::")
        # Give each sibling its own claim so both actually have rows left
        # to render with -- an unfiltered first sibling would otherwise
        # claim literally everything its parent has, leaving the second
        # with nothing (see the waterfall semantics in the module
        # docstring): a real "each gets its own view" scenario needs them
        # to be distinguishable, not both left as an open-ended catch-all.
        at.multiselect(key=f"{c0}claim_filter_keys").set_value(["street_norm"]).run(timeout=60)
        at.multiselect(key=f"{c1}claim_filter_keys").set_value(["street_norm"]).run(timeout=60)
        _batch_set(at, {
            f"{c0}claim_filter_values::street_norm": ["Preflop"],
            f"{c1}claim_filter_values::street_norm": ["Flop"],
        })

        assert not at.exception
        charts = at.get("plotly_chart")
        # root's own + one per sub-strategy.
        assert len(charts) == 3
        assert len({c.id for c in charts}) == 3
        assert any(c.id.endswith("root::graph_chart::hole_suited") for c in charts)
        assert any(c.id.endswith(f"{c0}graph_chart::hole_suited") for c in charts)
        assert any(c.id.endswith(f"{c1}graph_chart::hole_suited") for c in charts)

    def test_substrategy_cross_dropdown_is_independent_of_roots_own(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::hole_suited").set_value("Graph")
        at.run(timeout=60)
        c0 = _add_substrategy(at, "root::")

        cross_widgets = {ms.key: ms for ms in at.multiselect if ms.key and "graph_cross::" in ms.key}
        assert set(cross_widgets) == {"root::graph_cross::hole_suited", f"{c0}graph_cross::hole_suited"}

        cross_widgets["root::graph_cross::hole_suited"].set_value(["hand_category_norm"])
        at.run(timeout=60)

        assert not at.exception
        assert cross_widgets[f"{c0}graph_cross::hole_suited"].key in {
            ms.key for ms in at.multiselect if ms.key and "graph_cross::" in ms.key
        }
        substrategy_cross = at.multiselect(key=f"{c0}graph_cross::hole_suited")
        assert substrategy_cross.value == []  # unaffected by root's own selection


class TestSubStrategies:
    """The new priority-ordered "sub-strategy" mechanic (see
    _render_substrategy's own docstring): "Add sub-strategy" creates an
    ordered child; each child's own claim filter takes rows away from its
    parent before the next sibling (or the parent's own default behaviour)
    ever sees them."""

    def test_add_substrategy_button_exists_at_root(self, synthetic_checkpoint):
        at = _run_app()
        assert at.button(key="root::add_substrategy")

    def test_adding_a_substrategy_renders_a_heading_and_its_own_controls(self, synthetic_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")
        assert not at.exception
        assert len(at.subheader) == 1
        assert "no filter set yet" in at.subheader[0].value
        assert at.multiselect(key=f"{c0}claim_filter_keys")
        assert at.multiselect(key=f"{c0}split_by")
        assert at.button(key=f"{c0}add_substrategy")
        assert at.button(key=f"{c0}remove")
        assert at.button(key=f"{c0}move_up")
        assert at.button(key=f"{c0}move_down")

    def test_unclaimed_substrategy_sees_every_row_its_parent_has(self, synthetic_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")
        assert not at.exception
        assert "(n=200)" in at.subheader[0].value

    def test_claim_filter_narrows_the_heading_count_and_names_the_feature(self, synthetic_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")
        at.multiselect(key=f"{c0}claim_filter_keys").set_value(["street_norm"]).run(timeout=60)
        observed = at.multiselect(key=f"{c0}claim_filter_values::street_norm").value
        _batch_set(at, {f"{c0}claim_filter_values::street_norm": observed[:1]})

        assert not at.exception
        assert observed[0] in at.subheader[0].value
        assert "no filter set yet" not in at.subheader[0].value

    def test_second_sibling_only_sees_rows_the_first_did_not_claim(self, synthetic_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")
        c1 = _add_substrategy(at, "root::")

        _batch_set(at, {
            f"{c0}claim_filter_keys": ["street_norm"],
            f"{c1}claim_filter_keys": ["street_norm"],
        })
        _batch_set(at, {
            f"{c0}claim_filter_values::street_norm": ["Preflop"],
            f"{c1}claim_filter_values::street_norm": ["Flop"],
        })

        assert not at.exception
        subheaders = [h.value for h in at.subheader]
        c0_n = int(re.search(r"n=([\d,]+)", subheaders[0]).group(1).replace(",", ""))
        c1_n = int(re.search(r"n=([\d,]+)", subheaders[1]).group(1).replace(",", ""))
        assert c0_n > 0
        assert c1_n > 0
        # Preflop and Flop are disjoint street buckets -- the second
        # sibling's own count reflects rows the first one never touched,
        # not a re-filter of the parent's full 200.
        assert c0_n + c1_n <= 200

    def test_no_rows_match_shows_a_warning(self, synthetic_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")
        at.multiselect(key=f"{c0}claim_filter_keys").set_value(["street_norm"]).run(timeout=60)
        _batch_set(at, {f"{c0}claim_filter_values::street_norm": []})

        assert not at.exception
        assert any("No rows match this sub-strategy's own claim filters." in w.value for w in at.warning)

    def test_remove_deletes_the_substrategy(self, synthetic_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")
        _add_substrategy(at, "root::")
        assert len(at.session_state["substrategy_children"]["root::"]) == 2

        at.button(key=f"{c0}remove").click().run(timeout=60)

        assert not at.exception
        assert len(at.session_state["substrategy_children"]["root::"]) == 1
        assert not any(b.key == f"{c0}add_substrategy" for b in at.button)

    def test_move_up_swaps_priority_order(self, synthetic_checkpoint):
        at = _run_app()
        _add_substrategy(at, "root::")
        _add_substrategy(at, "root::")
        ids_before = list(at.session_state["substrategy_children"]["root::"])

        second_prefix = f"root::substrategy_{ids_before[1]}::"
        at.button(key=f"{second_prefix}move_up").click().run(timeout=60)

        assert not at.exception
        ids_after = list(at.session_state["substrategy_children"]["root::"])
        assert ids_after == [ids_before[1], ids_before[0]]

    def test_move_down_swaps_priority_order(self, synthetic_checkpoint):
        at = _run_app()
        _add_substrategy(at, "root::")
        _add_substrategy(at, "root::")
        ids_before = list(at.session_state["substrategy_children"]["root::"])

        first_prefix = f"root::substrategy_{ids_before[0]}::"
        at.button(key=f"{first_prefix}move_down").click().run(timeout=60)

        assert not at.exception
        ids_after = list(at.session_state["substrategy_children"]["root::"])
        assert ids_after == [ids_before[1], ids_before[0]]

    def test_nested_substrategy_gets_its_own_further_claim(self, synthetic_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")
        grandchild = _add_substrategy(at, c0)

        assert not at.exception
        assert at.multiselect(key=f"{grandchild}claim_filter_keys")
        assert len(at.subheader) == 1  # only c0 (level 0) -- grandchild nests one level deeper
        # The grandchild's own "no filter set yet" heading renders as a
        # markdown heading (level 1), not another subheader.
        assert any("no filter set yet" in m.value for m in at.markdown)


class TestSplitBy:
    """Each sub-strategy's own 1-2 "Split By" features define its
    prominent, implementable table+graph and %SHAP-explained figure (see
    _render_substrategy)."""

    def test_root_split_by_comes_from_the_sidebar(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::hand_category_norm").set_value("Split By")
        at.run(timeout=60)
        assert not at.exception
        # No local Split By widget at root -- the sidebar role assignment
        # already is root's own choice.
        assert not any(ms.key == "root::split_by" for ms in at.multiselect)
        assert len(at.dataframe) > 0

    def test_substrategy_has_its_own_local_split_by_widget(self, synthetic_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")
        assert at.multiselect(key=f"{c0}split_by")

    def test_substrategy_split_by_defaults_to_inherit_the_sidebars_pick(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::hand_category_norm").set_value("Split By")
        at.run(timeout=60)
        c0 = _add_substrategy(at, "root::")
        assert at.multiselect(key=f"{c0}split_by").value == ["hand_category_norm"]

    def test_picking_a_local_split_by_renders_its_own_table_and_chart(self, synthetic_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")
        # The prominent table (_render_table) renders unconditionally, even
        # with 0 Split By features picked yet (a single "All rows" summary)
        # -- so the dataframe *count* doesn't change, only its shape.
        dataframes_before = len(at.dataframe)
        charts_before = len(at.get("plotly_chart"))

        at.multiselect(key=f"{c0}split_by").set_value(["street_norm"]).run(timeout=60)

        assert not at.exception
        assert len(at.dataframe) == dataframes_before  # still just summary + counts, now split by street_norm
        assert len(at.get("plotly_chart")) == charts_before + 1  # one feature -> a line chart newly appears

    def test_two_split_by_features_render_heatmaps_instead_of_a_line_chart(self, synthetic_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")
        at.multiselect(key=f"{c0}split_by").set_value(["street_norm", "hole_suited"]).run(timeout=60)

        assert not at.exception
        charts = [c for c in at.get("plotly_chart") if f"{c0}splitby_heat::" in c.id]
        assert len(charts) == 4  # one per collapsed action group

    def test_shap_explained_metric_appears_once_split_by_is_set(self, synthetic_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")
        assert not any(m.label == "SHAP explained by Split By features" for m in at.metric)

        at.multiselect(key=f"{c0}split_by").set_value(["street_norm"]).run(timeout=60)

        assert not at.exception
        metrics = [m for m in at.metric if m.label == "SHAP explained by Split By features"]
        assert len(metrics) == 1
        assert metrics[0].value.endswith("%")

    def test_no_split_by_shows_a_hint_instead_of_a_metric(self, synthetic_checkpoint):
        at = _run_app()
        _add_substrategy(at, "root::")
        assert not at.exception
        assert any("Pick 1-2 Split By features" in c.value for c in at.caption)
        assert not any(m.label == "SHAP explained by Split By features" for m in at.metric)


_HOLE_HAND_GRID_FEATURE_KEYS = ("street_norm", "hole_hand_grid_x_norm", "hole_hand_grid_y_norm")


def _make_hole_hand_grid_checkpoint(path: str, rng: np.random.Generator, all_postflop: bool = False) -> None:
    """A checkpoint whose feature set includes the Exact Hole Hand pair
    alongside one ordinary feature (street_norm), correlated the way real
    training data is: preflop rows (street_norm pinned to exactly 0.0,
    "Preflop") carry a real grid position -- all pinned to the same cell,
    AA, so the resulting heatmaps have exactly one populated cell to check
    -- postflop rows (street_norm pinned to exactly 1.0, "River") carry the
    masked sentinel instead. `all_postflop` makes every row postflop, for
    testing the no-preflop-rows-anywhere case."""
    feature_dim = len(cfr_features.feature_indices(_HOLE_HAND_GRID_FEATURE_KEYS))
    street_idx = _HOLE_HAND_GRID_FEATURE_KEYS.index("street_norm")
    x_idx = _HOLE_HAND_GRID_FEATURE_KEYS.index("hole_hand_grid_x_norm")
    y_idx = _HOLE_HAND_GRID_FEATURE_KEYS.index("hole_hand_grid_y_norm")
    net = cfr_networks.AdvantageNet(input_dim=feature_dim, hidden_sizes=(8, 8))
    net_config = cfr_networks.AdvantageNetConfig(
        feature_keys=_HOLE_HAND_GRID_FEATURE_KEYS, hidden_sizes=(8, 8), table_size=3,
    )
    cfr_networks.save(net, net_config, path)

    num_samples = 100
    reservoir = cfr_reservoir.ReservoirBuffer(
        capacity=num_samples, feature_dim=feature_dim, num_actions=strategy.NUM_ACTION_CATEGORIES, rng=rng,
    )
    for i in range(num_samples):
        feats = rng.random(feature_dim).astype(np.float32)
        if not all_postflop and i % 2 == 0:
            feats[street_idx] = 0.0  # Preflop
            feats[x_idx] = 0.0  # AA
            feats[y_idx] = 0.0
        else:
            feats[street_idx] = 1.0  # River
            feats[x_idx] = -1.0  # masked, as if postflop
            feats[y_idx] = -1.0
        regrets = rng.normal(size=strategy.NUM_ACTION_CATEGORIES).astype(np.float32)
        legal_mask = rng.random(strategy.NUM_ACTION_CATEGORIES) > 0.3
        legal_mask[strategy.ACTION_CALL] = True
        reservoir.add(feats, regrets, legal_mask, float(rng.integers(1, 10)))
    reservoir.save(path)


@pytest.fixture(scope="module")
def _hole_hand_grid_checkpoint_path(tmp_path_factory) -> str:
    path = os.path.join(str(tmp_path_factory.mktemp("cfr_explorer_hole_hand_grid_checkpoint")), "checkpoint")
    _make_hole_hand_grid_checkpoint(path, np.random.default_rng(0))
    return path


@pytest.fixture
def hole_hand_grid_checkpoint(_hole_hand_grid_checkpoint_path, monkeypatch):
    monkeypatch.setenv("CFR_EXPLORER_CHECKPOINT_PATH", _hole_hand_grid_checkpoint_path)
    return _hole_hand_grid_checkpoint_path


@pytest.fixture(scope="module")
def _all_preflop_hole_hand_grid_checkpoint_path(tmp_path_factory) -> str:
    """Every row preflop -- so a sub-strategy needs no claim filter at all
    for Exact Hole Hand's 100%-preflop Split By requirement to already
    hold, letting tests exercise that path without also having to chain a
    claim-filter interaction in the same test (see _batch_set's docstring
    for why that combination is worth avoiding in this file)."""
    path = os.path.join(str(tmp_path_factory.mktemp("cfr_explorer_all_preflop_hole_hand_grid")), "checkpoint")
    rng = np.random.default_rng(0)
    feature_dim = len(cfr_features.feature_indices(_HOLE_HAND_GRID_FEATURE_KEYS))
    net = cfr_networks.AdvantageNet(input_dim=feature_dim, hidden_sizes=(8, 8))
    net_config = cfr_networks.AdvantageNetConfig(
        feature_keys=_HOLE_HAND_GRID_FEATURE_KEYS, hidden_sizes=(8, 8), table_size=3,
    )
    cfr_networks.save(net, net_config, path)
    num_samples = 100
    reservoir = cfr_reservoir.ReservoirBuffer(
        capacity=num_samples, feature_dim=feature_dim, num_actions=strategy.NUM_ACTION_CATEGORIES, rng=rng,
    )
    x_idx = _HOLE_HAND_GRID_FEATURE_KEYS.index("hole_hand_grid_x_norm")
    y_idx = _HOLE_HAND_GRID_FEATURE_KEYS.index("hole_hand_grid_y_norm")
    street_idx = _HOLE_HAND_GRID_FEATURE_KEYS.index("street_norm")
    for _ in range(num_samples):
        feats = rng.random(feature_dim).astype(np.float32)
        feats[street_idx] = 0.0
        feats[x_idx] = rng.integers(0, 13) / 12.0
        feats[y_idx] = rng.integers(0, 13) / 12.0
        regrets = rng.normal(size=strategy.NUM_ACTION_CATEGORIES).astype(np.float32)
        legal_mask = rng.random(strategy.NUM_ACTION_CATEGORIES) > 0.3
        legal_mask[strategy.ACTION_CALL] = True
        reservoir.add(feats, regrets, legal_mask, float(rng.integers(1, 10)))
    reservoir.save(path)
    return path


@pytest.fixture
def all_preflop_hole_hand_grid_checkpoint(_all_preflop_hole_hand_grid_checkpoint_path, monkeypatch):
    monkeypatch.setenv("CFR_EXPLORER_CHECKPOINT_PATH", _all_preflop_hole_hand_grid_checkpoint_path)
    return _all_preflop_hole_hand_grid_checkpoint_path


class TestExactHoleHand:
    def test_appears_in_sidebar_with_restricted_roles(self, hole_hand_grid_checkpoint):
        at = _run_app()
        assert not at.exception
        sb = at.sidebar.selectbox(key="role::hole_hand_grid_x_norm")
        assert list(sb.proto.options) == ["Unused", "Split By", "Graph"]
        assert not sb.proto.disabled  # some preflop rows are in view by default

    def test_second_axis_gets_no_selectbox_of_its_own(self, hole_hand_grid_checkpoint):
        at = _run_app()
        role_keys = {sb.key for sb in at.sidebar.selectbox if sb.key}
        assert "role::hole_hand_grid_y_norm" not in role_keys
        assert "role::street_norm" in role_keys  # the ordinary feature still gets its own

    def test_not_displayed_by_default(self, hole_hand_grid_checkpoint):
        at = _run_app()
        assert not at.exception
        assert not at.get("plotly_chart")
        assert not any(h.value == "Graphs" for h in at.header)

    def test_marking_as_graph_renders_heatmaps_in_the_graphs_section(self, hole_hand_grid_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::hole_hand_grid_x_norm").set_value("Graph")
        at.run(timeout=60)

        assert not at.exception
        assert any(h.value == "Graphs" for h in at.header)
        charts = [c for c in at.get("plotly_chart") if "graph_chart::hole_hand_grid_x_norm::" in c.id]
        assert len(charts) == 4
        titles = sorted(json.loads(c.proto.spec)["layout"]["title"]["text"] for c in charts)
        assert titles == sorted(f"{label} rate" for label in _COLLAPSED_LABELS)
        # No "cross with other features" control -- doesn't make sense for an already-2D feature.
        assert not any(ms.key and ms.key.endswith("graph_cross::hole_hand_grid_x_norm") for ms in at.multiselect)

    def test_axes_run_ace_to_deuce_with_ace_at_top_left(self, hole_hand_grid_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::hole_hand_grid_x_norm").set_value("Graph")
        at.run(timeout=60)

        chart = next(c for c in at.get("plotly_chart") if "graph_chart::hole_hand_grid_x_norm::" in c.id)
        spec = json.loads(chart.proto.spec)
        assert spec["data"][0]["x"] == list("AKQJT98765432")
        assert spec["data"][0]["y"] == list("23456789TJQKA")  # reversed -- last entry (A) renders at the top
        # AA sits at the last row (top, post-reversal), first column (left).
        assert spec["data"][0]["text"][-1][0] == "AA"

    def test_other_features_dont_offer_it_as_a_cross_target(self, hole_hand_grid_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::street_norm").set_value("Graph")
        at.run(timeout=60)

        cross = at.multiselect(key="root::graph_cross::street_norm")
        assert "Exact Hole Hand" not in cross.options

    def test_absent_when_checkpoint_lacks_the_feature(self, synthetic_checkpoint):
        at = _run_app()
        assert not at.exception
        role_keys = {sb.key for sb in at.sidebar.selectbox if sb.key}
        assert "role::hole_hand_grid_x_norm" not in role_keys

    def test_disabled_when_global_filter_excludes_every_preflop_row(self, hole_hand_grid_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::street_norm").set_value("Filter")
        at.run(timeout=60)
        at.sidebar.multiselect(key="filter::street_norm").set_value(["River"])
        at.run(timeout=60)

        assert not at.exception
        sb = at.sidebar.selectbox(key="role::hole_hand_grid_x_norm")
        assert sb.proto.disabled

    def test_absent_when_every_row_is_postflop(self, tmp_path_factory, monkeypatch):
        path = os.path.join(str(tmp_path_factory.mktemp("cfr_explorer_all_postflop")), "checkpoint")
        _make_hole_hand_grid_checkpoint(path, np.random.default_rng(1), all_postflop=True)
        monkeypatch.setenv("CFR_EXPLORER_CHECKPOINT_PATH", path)

        at = _run_app()
        assert not at.exception
        sb = at.sidebar.selectbox(key="role::hole_hand_grid_x_norm")
        assert sb.proto.disabled

    def test_selectable_as_split_by(self, hole_hand_grid_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")
        splitby = at.multiselect(key=f"{c0}split_by")
        assert any("Exact Hole Hand" in o for o in splitby.options)

    def test_split_by_fills_both_slots_and_warns_about_a_dropped_co_selection(self, hole_hand_grid_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")
        at.multiselect(key=f"{c0}split_by").set_value(["hole_hand_grid_x_norm", "street_norm"]).run(timeout=60)

        assert not at.exception
        # The widget itself keeps showing exactly what was clicked, but the
        # *resolved* Split By used for rendering treats Exact Hole Hand as
        # the sole pick -- confirmed structurally: this is postflop-mixed
        # data, so if street_norm had also been kept as a real second Split
        # By feature, _render_table's ordinary pivot would render instead
        # of erroring out with the "needs 100% preflop" warning.
        assert at.multiselect(key=f"{c0}split_by").value == ["hole_hand_grid_x_norm", "street_norm"]
        assert any("fills both Split By slots by itself" in c.value for c in at.caption)
        assert any("100% preflop" in w.value for w in at.warning)

    def test_split_by_renders_heatmaps_when_the_substrategy_is_already_100_percent_preflop(
        self, all_preflop_hole_hand_grid_checkpoint,
    ):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")
        at.multiselect(key=f"{c0}split_by").set_value(["hole_hand_grid_x_norm"]).run(timeout=60)

        assert not at.exception
        charts = [c for c in at.get("plotly_chart") if f"{c0}splitby_grid::" in c.id]
        assert len(charts) == 4
        assert not any("100% preflop" in w.value for w in at.warning)

    def test_split_by_warns_instead_of_rendering_when_rows_are_mixed_streets(self, hole_hand_grid_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")
        at.multiselect(key=f"{c0}split_by").set_value(["hole_hand_grid_x_norm"]).run(timeout=60)

        assert not at.exception
        charts = [c for c in at.get("plotly_chart") if f"{c0}splitby_grid::" in c.id]
        assert len(charts) == 0
        assert any("100% preflop" in w.value for w in at.warning)

    def test_root_split_by_from_sidebar_also_fills_both_slots(self, hole_hand_grid_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::hole_hand_grid_x_norm").set_value("Split By")
        at.sidebar.selectbox(key="role::street_norm").set_value("Split By")
        at.run(timeout=60)

        assert not at.exception
        assert any("fills both of Split By's slots by itself" in e.value for e in at.error)
