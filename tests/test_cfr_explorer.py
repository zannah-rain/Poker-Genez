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
        at.sidebar.selectbox(key="role::hole_suited").set_value("Table split")
        at.sidebar.selectbox(key="role::hand_category_norm").set_value("Table split")
        at.run(timeout=60)
        assert not at.exception
        assert len(at.error) >= 1

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

    def test_one_divider_per_group_heading_not_between_table_and_graphs(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::street_norm").set_value("Group split")
        at.sidebar.selectbox(key="role::hole_suited").set_value("Graph")
        at.run(timeout=60)

        assert not at.exception
        # One divider per observed street_norm value, plus the sidebar's own
        # divider above its collapse toggle -- none left over between a
        # group's table and its "Graphs" section, which is where a divider
        # used to sit.
        assert len(at.divider) == 4 + 1

    def test_no_divider_at_all_without_any_group_split(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::hole_suited").set_value("Graph")
        at.run(timeout=60)

        assert not at.exception
        assert len(at.divider) == 1  # just the sidebar's own divider -- no group headings to put one above


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
        at.multiselect(key="graph_cross::street_norm").set_value(["hand_category_norm"])
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

        cross = at.multiselect(key="graph_cross::street_norm")
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
        at.multiselect(key="graph_cross::street_norm").set_value(["hand_category_norm"])
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

    def test_ungrouped_graphs_section_uses_the_prominent_header(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::hole_suited").set_value("Graph")
        at.run(timeout=60)

        assert not at.exception
        assert any(h.value == "Graphs" for h in at.header)
        assert not any("Graphs" in m.value for m in at.markdown)

    def test_grouped_graphs_heading_is_one_level_below_its_group_heading(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::street_norm").set_value("Group split")
        at.sidebar.selectbox(key="role::hole_suited").set_value("Graph")
        at.run(timeout=60)

        assert not at.exception
        # The group heading itself is a subheader (level 0); "Graphs" for
        # that same group should render one level smaller, as a markdown
        # heading, not compete with it as another st.header/subheader.
        assert not any(h.value == "Graphs" for h in at.header)
        assert not any(h.value == "Graphs" for h in at.subheader)
        graphs_headings = [m.value for m in at.markdown if "Graphs" in m.value]
        assert len(graphs_headings) == 4
        assert all(h == "#### Graphs" for h in graphs_headings)  # one level below level-0's implicit h3

    def test_group_split_gives_each_group_its_own_graph(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::street_norm").set_value("Group split")
        at.sidebar.selectbox(key="role::hole_suited").set_value("Graph")
        at.run(timeout=60)

        assert not at.exception
        charts = at.get("plotly_chart")
        assert len(charts) == 4  # one street_norm value observed per bucket, times one graphed feature
        assert len({c.id for c in charts}) == 4  # every group's chart gets its own widget key, none collide

    def test_group_split_graph_only_reflects_that_groups_own_rows(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::street_norm").set_value("Group split")
        at.sidebar.selectbox(key="role::hole_suited").set_value("Graph")
        at.run(timeout=60)

        assert not at.exception
        # Recomputing "All rows" (no group split) per street_norm value directly
        # from the app's own rendered per-street summary tables would just
        # restate this test's own assumption, so instead confirm structurally
        # that each street's chart is independently keyed/rendered (the values
        # themselves come from the same regret-matching net either way) --
        # each street's graph is a distinct chart tied to that street's own
        # widget key, not one shared chart repeated four times.
        specs = [json.loads(c.proto.spec) for c in at.get("plotly_chart")]
        y_values = [tuple(trace["y"].get("bdata", trace.get("y")) for trace in spec["data"]) for spec in specs]
        assert len(set(y_values)) == len(y_values)  # no two streets produced byte-identical trace data

    def test_group_splits_cross_dropdown_is_independent_per_group(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::street_norm").set_value("Group split")
        at.sidebar.selectbox(key="role::hole_suited").set_value("Graph")
        at.run(timeout=60)

        cross_widgets = [ms for ms in at.multiselect if ms.key and "graph_cross::" in ms.key]
        assert len(cross_widgets) == 4
        assert len({ms.key for ms in cross_widgets}) == 4

        cross_widgets[0].set_value(["hand_category_norm"])
        at.run(timeout=60)

        assert not at.exception
        assert len(at.get("plotly_chart")) == 4 + 4  # 4 line charts + 4 heatmaps for just the one group crossed
        other_cross_widgets = [ms for ms in at.multiselect if ms.key and "graph_cross::" in ms.key][1:]
        assert all(ms.value == [] for ms in other_cross_widgets)  # unaffected by the first group's own selection


class TestPerGroupControls:
    def test_four_add_dropdowns_appear_under_every_group_heading(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::street_norm").set_value("Group split")
        at.run(timeout=60)

        assert not at.exception
        for suffix in ("local_graph", "local_table", "local_filter", "local_subgroup"):
            # Excludes the global::-prefixed row (see
            # test_global_level_also_gets_the_four_add_dropdowns below) --
            # that one sits above every heading, not under one.
            widgets = [
                ms for ms in at.multiselect
                if ms.key and ms.key.endswith(f"::{suffix}") and not ms.key.startswith("global::")
            ]
            assert len(widgets) == 4  # one street_norm value observed per bucket
            assert len({ms.key for ms in widgets}) == 4  # every group's own widget, none collide

    def test_global_level_also_gets_the_four_add_dropdowns(self, synthetic_checkpoint):
        # The global/top-level view (no grouping active at all) should get
        # the exact same Add graph/table/filter/subgroup row every subgroup
        # heading gets, not a bare table+graphs with no way to add to it.
        at = _run_app()
        assert not at.exception
        for suffix in ("local_graph", "local_table", "local_filter", "local_subgroup"):
            widgets = [ms for ms in at.multiselect if ms.key == f"global::{suffix}"]
            assert len(widgets) == 1

    def test_dropdown_options_exclude_the_group_split_feature(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::street_norm").set_value("Group split")
        at.run(timeout=60)

        local_widgets = [ms for ms in at.multiselect if ms.key and ms.key.startswith("street_norm=")]
        assert local_widgets
        assert all("Betting Street" not in ms.options for ms in local_widgets)
        assert all(set(ms.options) == {"Suited Hole Cards", "Hand Strength Tier"} for ms in local_widgets)

    def test_add_graph_for_one_group_only_renders_a_chart_there(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::street_norm").set_value("Group split")
        at.run(timeout=60)

        add_graph_widgets = [ms for ms in at.multiselect if ms.key and ms.key.endswith("::local_graph")]
        add_graph_widgets[0].set_value(["hole_suited"])
        at.run(timeout=60)

        assert not at.exception
        charts = at.get("plotly_chart")
        assert len(charts) == 1
        assert "local::graph_chart::hole_suited" in charts[0].id
        assert add_graph_widgets[0].key.removesuffix("local_graph") in charts[0].id

    def test_add_table_for_one_group_renders_a_labeled_extra_table(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::street_norm").set_value("Group split")
        at.run(timeout=60)
        dataframes_before = len(at.dataframe)

        add_table_widgets = [ms for ms in at.multiselect if ms.key and ms.key.endswith("::local_table")]
        add_table_widgets[0].set_value(["hand_category_norm"])
        at.run(timeout=60)

        assert not at.exception
        assert len(at.dataframe) == dataframes_before + 2  # _render_table's own summary + sample-counts dataframes
        assert any("Extra table: Hand Strength Tier" in c.value for c in at.caption)

    def test_add_filter_narrows_just_that_groups_own_rows(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::street_norm").set_value("Group split")
        at.run(timeout=60)

        add_filter_widgets = [ms for ms in at.multiselect if ms.key and ms.key.endswith("::local_filter")]
        group_prefix = add_filter_widgets[0].key.removesuffix("local_filter")
        add_filter_widgets[0].set_value(["hole_suited"])
        at.run(timeout=60)

        values_widget = at.multiselect(key=f"{group_prefix}local_filter_values::hole_suited")
        assert set(values_widget.value) == {"Not Suited Hole Cards", "Suited Hole Cards"}  # defaults to every observed value

        values_widget.set_value([])  # narrow to nothing, matching the app's existing "no rows" handling
        at.run(timeout=60)

        assert not at.exception
        assert any("No rows match this group's own local filters." in w.value for w in at.warning)

    def test_add_subgroup_only_splits_that_one_group_further(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::street_norm").set_value("Group split")
        at.run(timeout=60)

        # Excludes global::local_subgroup -- picking that one would apply
        # the split to every street_norm value at once, not just this one
        # group (see test_global_level_also_gets_the_four_add_dropdowns).
        add_subgroup_widgets = [
            ms for ms in at.multiselect
            if ms.key and ms.key.endswith("::local_subgroup") and ms.key.startswith("street_norm=")
        ]
        add_subgroup_widgets[0].set_value(["hand_category_norm"])
        at.run(timeout=60)

        assert not at.exception
        # Nested headings render as markdown (one level deeper than the
        # top-level subheader -- see _render_group_heading) and only for
        # the one group that got the subgroup split.
        assert len(at.markdown) > 0
        subheaders = [h.value for h in at.subheader]
        assert len(subheaders) == 4  # still exactly one top-level heading per street_norm value

    def test_subgroup_key_excluded_from_the_nested_groups_own_dropdowns(self, synthetic_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::street_norm").set_value("Group split")
        at.run(timeout=60)

        # Excludes global::local_subgroup -- see
        # test_add_subgroup_only_splits_that_one_group_further.
        add_subgroup_widgets = [
            ms for ms in at.multiselect
            if ms.key and ms.key.endswith("::local_subgroup") and ms.key.startswith("street_norm=")
        ]
        chosen_group_prefix = add_subgroup_widgets[0].key.removesuffix("local_subgroup")
        add_subgroup_widgets[0].set_value(["hand_category_norm"])
        at.run(timeout=60)

        nested_widgets = [
            ms for ms in at.multiselect
            if ms.key and ms.key.startswith(f"{chosen_group_prefix}hand_category_norm=") and ms.key.endswith("::local_graph")
        ]
        assert nested_widgets
        assert all("Hand Strength Tier" not in ms.options for ms in nested_widgets)
        assert all(ms.options == ["Suited Hole Cards"] for ms in nested_widgets)

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


class TestExactHoleHand:
    def test_appears_in_sidebar_with_restricted_roles(self, hole_hand_grid_checkpoint):
        at = _run_app()
        assert not at.exception
        sb = at.sidebar.selectbox(key="role::hole_hand_grid_x_norm")
        assert list(sb.proto.options) == ["Unused", "Graph"]
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

        cross = at.multiselect(key="graph_cross::street_norm")
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

    def test_group_split_gives_each_group_its_own_view(self, hole_hand_grid_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::street_norm").set_value("Group split")
        at.sidebar.selectbox(key="role::hole_hand_grid_x_norm").set_value("Graph")
        at.run(timeout=60)

        assert not at.exception
        # The Preflop group has real grid data -- 4 heatmaps.
        preflop_charts = [
            c for c in at.get("plotly_chart")
            if "street_norm=Preflop::" in c.id and "graph_chart::hole_hand_grid_x_norm::" in c.id
        ]
        assert len(preflop_charts) == 4
        # The River group is all masked -- no heatmaps, just the fallback caption.
        river_charts = [
            c for c in at.get("plotly_chart")
            if "street_norm=River::" in c.id and "graph_chart::hole_hand_grid_x_norm::" in c.id
        ]
        assert len(river_charts) == 0
        assert any("No preflop rows" in c.value for c in at.caption)

    def test_excluded_from_add_graph_in_a_subgroup_without_preflop_rows(self, hole_hand_grid_checkpoint):
        at = _run_app()
        at.sidebar.selectbox(key="role::street_norm").set_value("Group split")
        at.run(timeout=60)

        local_graph_widgets = {
            ms.key: ms for ms in at.multiselect if ms.key and ms.key.endswith("::local_graph")
        }
        preflop_widget = next(ms for key, ms in local_graph_widgets.items() if "street_norm=Preflop::" in key)
        river_widget = next(ms for key, ms in local_graph_widgets.items() if "street_norm=River::" in key)
        assert "Exact Hole Hand" in preflop_widget.options
        assert "Exact Hole Hand" not in river_widget.options
        # Never offered in the other 3 "Add ..." dropdowns, preflop or not.
        other_dropdown_suffixes = ("::local_table", "::local_filter", "::local_subgroup")
        other_widgets = [
            ms for ms in at.multiselect
            if ms.key and ms.key.endswith(other_dropdown_suffixes) and "street_norm=Preflop::" in ms.key
        ]
        assert other_widgets
        assert all("Exact Hole Hand" not in ms.options for ms in other_widgets)

    def test_absent_when_every_row_is_postflop(self, tmp_path_factory, monkeypatch):
        path = os.path.join(str(tmp_path_factory.mktemp("cfr_explorer_all_postflop")), "checkpoint")
        _make_hole_hand_grid_checkpoint(path, np.random.default_rng(1), all_postflop=True)
        monkeypatch.setenv("CFR_EXPLORER_CHECKPOINT_PATH", path)

        at = _run_app()
        assert not at.exception
        sb = at.sidebar.selectbox(key="role::hole_hand_grid_x_norm")
        assert sb.proto.disabled
