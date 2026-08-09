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


def _strip_shap_suffix(option: str) -> str:
    """A "Add filter/graph/table" or "Split By" dropdown option's plain
    feature label, with its "  (SHAP 0.0000)" suffix (see
    _render_substrategy) removed -- so tests can check which features are
    offered without hardcoding an exact importance value."""
    return re.sub(r"\s+\(SHAP [\d.]+\)$", "", option)


def _option_labels(widget) -> set[str]:
    """The plain feature labels a SHAP-suffixed multiselect widget
    currently offers (see _strip_shap_suffix) -- AppTest's own `.options`
    already reflects the widget's format_func output, not the raw
    feature-key identity."""
    return {_strip_shap_suffix(o) for o in widget.options}


def _shap_values_by_key(at: AppTest, key_prefix: str = "root::", feature_keys=_FEATURE_KEYS) -> dict[str, float]:
    """Reads the "(SHAP 0.0000)" values off `key_prefix`'s own Split By
    multiselect options -- computed over that node's current `default_df`
    (see _render_substrategy), so it reflects whatever claim filters are
    currently narrowing that node."""
    label_to_key = {cfr_features.feature_label(k): k for k in feature_keys}
    values: dict[str, float] = {}
    for option in at.multiselect(key=f"{key_prefix}split_by").options:
        match = re.match(r"^(.*)  \(SHAP ([\d.]+)\)$", option)
        if not match:
            continue
        label, shap = match.groups()
        if label in label_to_key:
            values[label_to_key[label]] = float(shap)
    return values


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

    def test_add_filter_lists_every_configured_feature(self, synthetic_checkpoint):
        at = _run_app()
        assert _option_labels(at.multiselect(key="root::claim_filter_keys")) == {
            cfr_features.feature_label(k) for k in _FEATURE_KEYS
        }

    def test_empty_reservoir_shows_warning_not_a_crash(self, empty_reservoir_checkpoint):
        at = _run_app()
        assert not at.exception
        assert len(at.warning) >= 1

    def test_missing_checkpoint_shows_error_not_a_crash(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CFR_EXPLORER_CHECKPOINT_PATH", os.path.join(str(tmp_path), "does_not_exist"))
        at = _run_app()
        assert not at.exception
        assert len(at.error) >= 1


class TestRootIsJustAnotherSubstrategy:
    """Root gets the exact same central controls as any sub-strategy added
    via "Add sub-strategy" -- no separate sidebar version of any of them
    (see the module docstring and _render_substrategy)."""

    def test_root_gets_the_same_controls_as_any_child(self, synthetic_checkpoint):
        at = _run_app()
        assert not at.exception
        assert at.multiselect(key="root::claim_filter_keys")
        assert at.multiselect(key="root::split_by")
        assert at.button(key="root::add_substrategy")
        assert at.multiselect(key="root::extra_graph")
        assert at.multiselect(key="root::extra_table")

    def test_root_has_no_remove_or_move_buttons(self, synthetic_checkpoint):
        at = _run_app()
        assert not any(b.key == "root::remove" for b in at.button)
        assert not any(b.key == "root::move_up" for b in at.button)
        assert not any(b.key == "root::move_down" for b in at.button)

    def test_root_heading_says_overall_strategy(self, synthetic_checkpoint):
        at = _run_app()
        assert any(h.value.startswith("Overall Strategy") for h in at.header)

    def test_root_filter_narrows_its_own_scope_and_heading(self, synthetic_checkpoint):
        at = _run_app()
        at.multiselect(key="root::claim_filter_keys").set_value(["street_norm"]).run(timeout=60)
        observed = at.multiselect(key="root::claim_filter_values::street_norm").value
        _batch_set(at, {"root::claim_filter_values::street_norm": observed[:1]})

        assert not at.exception
        header = next(h.value for h in at.header if h.value.startswith("Overall Strategy"))
        assert observed[0] in header

    def test_root_filter_yielding_zero_rows_shows_a_warning_and_stops_rendering(self, synthetic_checkpoint):
        at = _run_app()
        at.multiselect(key="root::claim_filter_keys").set_value(["hole_suited"]).run(timeout=60)
        _batch_set(at, {"root::claim_filter_values::hole_suited": []})

        assert not at.exception
        assert any("No rows match this sub-strategy's own claim filters." in w.value for w in at.warning)
        assert not any(ms.key == "root::split_by" for ms in at.multiselect)


class TestSubstrategyShapImportance:
    def test_narrowing_a_filter_changes_reported_shap_values(self, synthetic_checkpoint):
        at = _run_app()
        before = _shap_values_by_key(at)

        at.multiselect(key="root::claim_filter_keys").set_value(["street_norm"]).run(timeout=60)
        observed = at.multiselect(key="root::claim_filter_values::street_norm").value
        _batch_set(at, {"root::claim_filter_values::street_norm": observed[:1]})

        after = _shap_values_by_key(at)
        assert not at.exception
        assert before != after

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

        at.multiselect(key="root::claim_filter_keys").set_value(["street_norm"]).run(timeout=60)
        _batch_set(at, {"root::claim_filter_values::street_norm": ["Preflop"]})

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


class TestGraphs:
    def test_marking_a_feature_as_graph_renders_one_line_chart(self, synthetic_checkpoint):
        at = _run_app()
        at.multiselect(key="root::extra_graph").set_value(["street_norm"]).run(timeout=60)

        assert not at.exception
        charts = at.get("plotly_chart")
        assert len(charts) == 1
        assert charts[0].id.endswith("root::extra::graph_chart::street_norm")

    def test_no_graphs_section_when_nothing_is_added(self, synthetic_checkpoint):
        at = _run_app()
        assert not at.get("plotly_chart")
        assert not any("Graphs" in m.value for m in at.markdown)

    def test_crossing_with_another_feature_adds_four_heatmaps(self, synthetic_checkpoint):
        at = _run_app()
        at.multiselect(key="root::extra_graph").set_value(["street_norm"]).run(timeout=60)
        at.multiselect(key="root::extra::graph_cross::street_norm").set_value(["hand_category_norm"])
        at.run(timeout=60)

        assert not at.exception
        charts = at.get("plotly_chart")
        assert len(charts) == 5  # 1 line chart + 4 simplified-action heatmaps
        heat_ids = [c.id for c in charts if "graph_heat::street_norm::hand_category_norm::" in c.id]
        assert len(heat_ids) == 4

    def test_cross_dropdown_excludes_the_graphed_feature_itself(self, synthetic_checkpoint):
        at = _run_app()
        at.multiselect(key="root::extra_graph").set_value(["street_norm"]).run(timeout=60)

        cross = at.multiselect(key="root::extra::graph_cross::street_norm")
        assert "Betting Street" not in cross.options  # street_norm's own label -- can't cross a feature with itself
        assert set(cross.options) == {"Suited Hole Cards", "Hand Strength Tier"}

    def test_line_chart_has_one_trace_per_action_and_a_percent_y_axis(self, synthetic_checkpoint):
        at = _run_app()
        at.multiselect(key="root::extra_graph").set_value(["street_norm"]).run(timeout=60)

        spec = json.loads(at.get("plotly_chart")[0].proto.spec)
        assert len(spec["data"]) == strategy.NUM_ACTION_CATEGORIES
        assert spec["layout"]["yaxis"]["range"] == [0, 1]
        assert spec["layout"]["yaxis"]["tickformat"] == ".0%"

    def test_collapsing_actions_also_collapses_the_line_chart_to_four_traces(self, synthetic_checkpoint):
        at = _run_app()
        at.multiselect(key="root::extra_graph").set_value(["street_norm"]).run(timeout=60)
        at.sidebar.toggle[0].set_value(True)
        at.run(timeout=60)

        spec = json.loads(at.get("plotly_chart")[0].proto.spec)
        assert len(spec["data"]) == 4
        assert [trace["name"] for trace in spec["data"]] == list(_COLLAPSED_LABELS)

    def test_heatmaps_always_use_the_four_simplified_actions_regardless_of_toggle(self, synthetic_checkpoint):
        at = _run_app()
        at.multiselect(key="root::extra_graph").set_value(["street_norm"]).run(timeout=60)
        at.multiselect(key="root::extra::graph_cross::street_norm").set_value(["hand_category_norm"])
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

    def test_graphs_heading_is_a_subordinate_markdown_heading_at_root(self, synthetic_checkpoint):
        at = _run_app()
        at.multiselect(key="root::extra_graph").set_value(["hole_suited"]).run(timeout=60)

        assert not at.exception
        assert not any(h.value == "Graphs" for h in at.header)
        graphs_headings = [m.value for m in at.markdown if "Graphs" in m.value]
        assert len(graphs_headings) == 1
        assert graphs_headings[0] == "#### Graphs"

    def test_graphs_heading_is_the_same_subordinate_level_for_a_substrategy(self, synthetic_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")
        at.multiselect(key=f"{c0}extra_graph").set_value(["hole_suited"]).run(timeout=60)

        assert not at.exception
        graphs_headings = [m.value for m in at.markdown if "Graphs" in m.value]
        assert len(graphs_headings) == 1
        assert graphs_headings[0] == "#### Graphs"

    def test_each_substrategy_gets_its_own_independently_keyed_graph(self, synthetic_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")
        # A single, unclaimed child is enough to prove key independence --
        # not layering in a full disjoint-claim sibling waterfall here,
        # since re-touching several already-set claim-filter multiselects
        # in a later, unrelated batch is exactly the AppTest format_func
        # round-trip quirk _batch_set's docstring warns about (confirmed
        # during development: it silently reverts an earlier claim back to
        # unclaimed), which this test has no need to risk.
        _batch_set(at, {"root::extra_graph": ["hole_suited"], f"{c0}extra_graph": ["hole_suited"]})

        assert not at.exception
        charts = at.get("plotly_chart")
        assert len(charts) == 2
        assert len({c.id for c in charts}) == 2
        assert any(c.id.endswith("root::extra::graph_chart::hole_suited") for c in charts)
        assert any(c.id.endswith(f"{c0}extra::graph_chart::hole_suited") for c in charts)

    def test_substrategy_cross_dropdown_is_independent_of_roots_own(self, synthetic_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")
        _batch_set(at, {"root::extra_graph": ["hole_suited"], f"{c0}extra_graph": ["hole_suited"]})

        cross_widgets = {ms.key: ms for ms in at.multiselect if ms.key and "graph_cross::" in ms.key}
        assert set(cross_widgets) == {"root::extra::graph_cross::hole_suited", f"{c0}extra::graph_cross::hole_suited"}

        cross_widgets["root::extra::graph_cross::hole_suited"].set_value(["hand_category_norm"])
        at.run(timeout=60)

        assert not at.exception
        substrategy_cross = at.multiselect(key=f"{c0}extra::graph_cross::hole_suited")
        assert substrategy_cross.value == []  # unaffected by root's own selection


class TestSubStrategies:
    """The priority-ordered "sub-strategy" mechanic (see
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

    def test_root_has_its_own_local_split_by_widget(self, synthetic_checkpoint):
        at = _run_app()
        assert at.multiselect(key="root::split_by")
        assert len(at.dataframe) > 0

    def test_root_split_by_renders_a_dataframe(self, synthetic_checkpoint):
        at = _run_app()
        at.multiselect(key="root::split_by").set_value(["hand_category_norm"]).run(timeout=60)
        assert not at.exception
        assert len(at.dataframe) > 0

    def test_substrategy_has_its_own_local_split_by_widget(self, synthetic_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")
        assert at.multiselect(key=f"{c0}split_by")

    def test_substrategy_split_by_defaults_to_inherit_its_parents_pick(self, synthetic_checkpoint):
        at = _run_app()
        at.multiselect(key="root::split_by").set_value(["hand_category_norm"]).run(timeout=60)
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


class TestNavigation:
    """The sidebar's nested, clickable shorthand outline of the current
    sub-strategy tree (see _render_navigation), replacing the old sidebar
    feature dropdowns."""

    def test_anchor_renders_before_root_heading(self, synthetic_checkpoint):
        at = _run_app()
        assert not at.exception
        assert any(m.value == '<a name="root"></a>' for m in at.markdown)

    def test_root_link_present_by_default(self, synthetic_checkpoint):
        at = _run_app()
        nav = next(m for m in at.sidebar.markdown if "Overall Strategy" in m.value)
        assert "(#root)" in nav.value

    def test_child_gets_its_own_indented_nested_link(self, synthetic_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")
        child_id = c0.rsplit("substrategy_", 1)[1].rstrip(":")

        nav = next(m for m in at.sidebar.markdown if "Overall Strategy" in m.value)
        assert f"(#root-substrategy_{child_id})" in nav.value
        lines = nav.value.splitlines()
        root_line = next(l for l in lines if "(#root)" in l)
        child_line = next(l for l in lines if f"root-substrategy_{child_id}" in l)
        assert not root_line.startswith(" ")
        assert child_line.startswith("  ")

    def test_unclaimed_child_label_says_unclaimed(self, synthetic_checkpoint):
        at = _run_app()
        _add_substrategy(at, "root::")
        nav = next(m for m in at.sidebar.markdown if "Overall Strategy" in m.value)
        assert "Sub-strategy (unclaimed)" in nav.value

    def test_child_claim_shows_up_in_its_nav_label(self, synthetic_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")
        at.multiselect(key=f"{c0}claim_filter_keys").set_value(["street_norm"]).run(timeout=60)
        observed = at.multiselect(key=f"{c0}claim_filter_values::street_norm").value
        _batch_set(at, {f"{c0}claim_filter_values::street_norm": observed[:1]})

        nav = next(m for m in at.sidebar.markdown if "Overall Strategy" in m.value)
        assert observed[0] in nav.value

    def test_long_claim_description_gets_truncated_in_nav_but_not_in_heading(self, synthetic_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")
        at.multiselect(key=f"{c0}claim_filter_keys").set_value(["hand_category_norm"]).run(timeout=60)

        assert not at.exception
        observed = at.multiselect(key=f"{c0}claim_filter_values::hand_category_norm").value
        full_desc = f"Hand Strength Tier = {', '.join(observed)}"
        assert full_desc in at.subheader[0].value  # the real heading is never truncated

        nav = next(m for m in at.sidebar.markdown if "Overall Strategy" in m.value)
        child_line = next(l for l in nav.value.splitlines() if "substrategy_" in l)
        if len(full_desc) > 40:
            assert "…]" in child_line  # truncated with an ellipsis, right before the markdown link closes
            assert full_desc not in child_line
        else:
            assert full_desc in child_line


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
    def test_excluded_from_add_filter_options(self, hole_hand_grid_checkpoint):
        at = _run_app()
        assert not at.exception
        assert "Exact Hole Hand" not in _option_labels(at.multiselect(key="root::claim_filter_keys"))

    def test_available_as_split_by_when_preflop_rows_present(self, hole_hand_grid_checkpoint):
        at = _run_app()
        assert not at.exception
        assert "Exact Hole Hand" in _option_labels(at.multiselect(key="root::split_by"))

    def test_second_axis_collapses_into_a_single_split_by_option(self, hole_hand_grid_checkpoint):
        at = _run_app()
        options = [o for o in at.multiselect(key="root::split_by").options if "Exact Hole Hand" in o]
        assert len(options) == 1

    def test_not_displayed_by_default(self, hole_hand_grid_checkpoint):
        at = _run_app()
        assert not at.exception
        assert not at.get("plotly_chart")

    def test_marking_as_graph_renders_heatmaps_in_the_graphs_section(self, hole_hand_grid_checkpoint):
        at = _run_app()
        at.multiselect(key="root::extra_graph").set_value(["hole_hand_grid_x_norm"]).run(timeout=60)

        assert not at.exception
        charts = [c for c in at.get("plotly_chart") if "graph_chart::hole_hand_grid_x_norm::" in c.id]
        assert len(charts) == 4
        titles = sorted(json.loads(c.proto.spec)["layout"]["title"]["text"] for c in charts)
        assert titles == sorted(f"{label} rate" for label in _COLLAPSED_LABELS)
        # No "cross with other features" control -- doesn't make sense for an already-2D feature.
        assert not any(ms.key and ms.key.endswith("graph_cross::hole_hand_grid_x_norm") for ms in at.multiselect)

    def test_axes_run_ace_to_deuce_with_ace_at_top_left(self, hole_hand_grid_checkpoint):
        at = _run_app()
        at.multiselect(key="root::extra_graph").set_value(["hole_hand_grid_x_norm"]).run(timeout=60)

        chart = next(c for c in at.get("plotly_chart") if "graph_chart::hole_hand_grid_x_norm::" in c.id)
        spec = json.loads(chart.proto.spec)
        assert spec["data"][0]["x"] == list("AKQJT98765432")
        assert spec["data"][0]["y"] == list("23456789TJQKA")  # reversed -- last entry (A) renders at the top
        # AA sits at the last row (top, post-reversal), first column (left).
        assert spec["data"][0]["text"][-1][0] == "AA"

    def test_other_features_dont_offer_it_as_a_cross_target(self, hole_hand_grid_checkpoint):
        at = _run_app()
        at.multiselect(key="root::extra_graph").set_value(["street_norm"]).run(timeout=60)

        cross = at.multiselect(key="root::extra::graph_cross::street_norm")
        assert "Exact Hole Hand" not in cross.options

    def test_absent_when_checkpoint_lacks_the_feature(self, synthetic_checkpoint):
        at = _run_app()
        assert not at.exception
        assert "Exact Hole Hand" not in _option_labels(at.multiselect(key="root::split_by"))

    def test_add_graph_excludes_it_when_the_filtered_view_has_no_preflop_rows(self, hole_hand_grid_checkpoint):
        at = _run_app()
        at.multiselect(key="root::claim_filter_keys").set_value(["street_norm"]).run(timeout=60)
        _batch_set(at, {"root::claim_filter_values::street_norm": ["River"]})

        assert not at.exception
        assert "Exact Hole Hand" not in _option_labels(at.multiselect(key="root::extra_graph"))

    def test_add_graph_excludes_it_when_every_row_is_postflop(self, tmp_path_factory, monkeypatch):
        path = os.path.join(str(tmp_path_factory.mktemp("cfr_explorer_all_postflop")), "checkpoint")
        _make_hole_hand_grid_checkpoint(path, np.random.default_rng(1), all_postflop=True)
        monkeypatch.setenv("CFR_EXPLORER_CHECKPOINT_PATH", path)

        at = _run_app()
        assert not at.exception
        assert "Exact Hole Hand" not in _option_labels(at.multiselect(key="root::extra_graph"))

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

    def test_root_split_by_also_fills_both_slots(self, hole_hand_grid_checkpoint):
        at = _run_app()
        at.multiselect(key="root::split_by").set_value(["hole_hand_grid_x_norm", "street_norm"]).run(timeout=60)

        assert not at.exception
        assert any("fills both Split By slots by itself" in c.value for c in at.caption)


_GROUP_RELATIVE_FEATURE_KEYS = ("hole_hand_grid_x_norm", "hole_hand_grid_y_norm", "hole_suited")


def _make_group_relative_checkpoint(path: str, rng: np.random.Generator) -> None:
    """A checkpoint whose feature set includes Exact Hole Hand alongside
    `hole_suited`, deterministically derived from Exact Hole Hand's own
    grid position (upper-right of the diagonal, matching the real "suited"
    convention -- see features.hole_hand_grid_label) -- every row preflop,
    so Exact Hole Hand's Split By is available with no claim filter first
    needed. Regression coverage for "a sub-strategy's own SHAP view should
    assume its parent's own Split By grouping is already priced in" (see
    cfr_explorer._group_labels_for_rows/_render_substrategy): once a
    parent sub-strategy splits by Exact Hole Hand, hole_suited -- fully
    determined by it -- should score exactly 0.0000 SHAP for any
    sub-strategy underneath, instead of sharing credit for the same signal."""
    feature_dim = len(cfr_features.feature_indices(_GROUP_RELATIVE_FEATURE_KEYS))
    x_idx = _GROUP_RELATIVE_FEATURE_KEYS.index("hole_hand_grid_x_norm")
    y_idx = _GROUP_RELATIVE_FEATURE_KEYS.index("hole_hand_grid_y_norm")
    suited_idx = _GROUP_RELATIVE_FEATURE_KEYS.index("hole_suited")
    net = cfr_networks.AdvantageNet(input_dim=feature_dim, hidden_sizes=(16, 16))
    net_config = cfr_networks.AdvantageNetConfig(
        feature_keys=_GROUP_RELATIVE_FEATURE_KEYS, hidden_sizes=(16, 16), table_size=3,
    )
    cfr_networks.save(net, net_config, path)

    num_samples = 300
    reservoir = cfr_reservoir.ReservoirBuffer(
        capacity=num_samples, feature_dim=feature_dim, num_actions=strategy.NUM_ACTION_CATEGORIES, rng=rng,
    )
    for _ in range(num_samples):
        feats = rng.random(feature_dim).astype(np.float32)
        feats[x_idx] = rng.integers(0, 13) / 12.0
        feats[y_idx] = rng.integers(0, 13) / 12.0
        feats[suited_idx] = 1.0 if feats[x_idx] > feats[y_idx] else 0.0
        regrets = rng.normal(size=strategy.NUM_ACTION_CATEGORIES).astype(np.float32)
        legal_mask = rng.random(strategy.NUM_ACTION_CATEGORIES) > 0.3
        legal_mask[strategy.ACTION_CALL] = True
        reservoir.add(feats, regrets, legal_mask, float(rng.integers(1, 10)))
    reservoir.save(path)


@pytest.fixture(scope="module")
def _group_relative_checkpoint_path(tmp_path_factory) -> str:
    path = os.path.join(str(tmp_path_factory.mktemp("cfr_explorer_group_relative")), "checkpoint")
    _make_group_relative_checkpoint(path, np.random.default_rng(0))
    return path


@pytest.fixture
def group_relative_checkpoint(_group_relative_checkpoint_path, monkeypatch):
    monkeypatch.setenv("CFR_EXPLORER_CHECKPOINT_PATH", _group_relative_checkpoint_path)
    return _group_relative_checkpoint_path


class TestGroupRelativeShap:
    """A sub-strategy's own SHAP view assumes its *parent's* own Split By
    grouping is already priced in (see _group_labels_for_rows). In this
    checkpoint's synthetic data, Exact Hole Hand fully determines
    hole_suited, so once a parent splits by Exact Hole Hand, a child
    underneath it should show exactly 0.0000 SHAP for hole_suited instead
    of sharing credit for the same signal Exact Hole Hand already explains."""

    def test_hole_suited_shows_nonzero_shap_with_no_parent_grouping(self, group_relative_checkpoint):
        at = _run_app()
        assert not at.exception
        option = next(o for o in at.multiselect(key="root::split_by").options if o.startswith("Suited Hole Cards"))
        shap_value = float(re.search(r"SHAP ([\d.]+)\)", option).group(1))
        assert shap_value > 0.0

    def test_hole_suited_drops_to_exactly_zero_under_an_exact_hole_hand_parent_split(
        self, group_relative_checkpoint,
    ):
        at = _run_app()
        at.multiselect(key="root::split_by").set_value(["hole_hand_grid_x_norm"]).run(timeout=60)
        c0 = _add_substrategy(at, "root::")

        assert not at.exception
        option = next(o for o in at.multiselect(key=f"{c0}split_by").options if o.startswith("Suited Hole Cards"))
        assert option.endswith("(SHAP 0.0000)")

    def test_a_grandchild_under_an_unrelated_parent_split_keeps_a_nonzero_shap(
        self, group_relative_checkpoint,
    ):
        # Sanity check for the other direction: a child added under a
        # parent that has *not* split by Exact Hole Hand (root's own
        # default, unset Split By) should see hole_suited's ordinary,
        # nonzero SHAP -- confirming the zeroing above is specifically a
        # consequence of the parent's own grouping, not some unconditional
        # side effect of merely being a non-root node.
        at = _run_app()
        c0 = _add_substrategy(at, "root::")

        assert not at.exception
        option = next(o for o in at.multiselect(key=f"{c0}split_by").options if o.startswith("Suited Hole Cards"))
        shap_value = float(re.search(r"SHAP ([\d.]+)\)", option).group(1))
        assert shap_value > 0.0
