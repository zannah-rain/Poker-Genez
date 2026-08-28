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


@pytest.fixture(autouse=True)
def isolated_strategies_dir(tmp_path, monkeypatch) -> str:
    """Every test in this file gets its own empty scratch directory for
    cfr_explorer.py's own saved/autosaved strategy files (see
    _strategies_dir's own CFR_EXPLORER_STRATEGIES_DIR override) --
    autouse, since plenty of tests here share one on-disk checkpoint path
    via a module-scoped fixture (kept slow-to-build net inference results
    cached across tests -- see _synthetic_checkpoint_path), which would
    otherwise leak one test's own saved/autosaved strategy state into
    every other test that happens to reuse the same checkpoint path,
    entirely unrelated to what that other test is actually exercising."""
    strategies_dir = str(tmp_path / "explorer_strategies")
    monkeypatch.setenv("CFR_EXPLORER_STRATEGIES_DIR", strategies_dir)
    return strategies_dir


def _strip_variance_suffix(option: str) -> str:
    """A "Add filter/graph/table", "Split By", or "Cross with other
    features" dropdown option's plain feature label, with its own
    "  (NN% variance explained)" suffix (see _decision_variance_by_key/
    _render_substrategy/_render_graphs -- the one metric every dropdown in
    the app labels its options with) removed -- so tests can check which
    features are offered without hardcoding an exact value."""
    return re.sub(r"\s+\(\d+% variance explained\)$", "", option)


def _nav_button_css_class(key: str) -> str:
    """Reproduces Streamlit's own key -> CSS class sanitization (its `key`
    docs: "it will be used as a CSS class name prefixed with st-key-") --
    duplicated here (see _nav_button_css_class in cfr_explorer.py) rather
    than imported, since cfr_explorer.py is a script driven only via
    AppTest.from_file, not an importable module."""
    return "st-key-" + re.sub(r"[^a-zA-Z0-9_-]", "-", key.strip())


def _nav_indent_px(at, key_prefix: str) -> int:
    """The padding-left (px) the sidebar nav's injected <style> block (see
    _render_navigation_node) assigns to the button for `key_prefix`. Read
    off `at.get("html")` rather than `at.sidebar.get("html")` -- Streamlit
    routes style-only `st.html` content to a separate internal "event"
    container instead of the block it was actually called on, so it never
    shows up nested under the sidebar's own element tree."""
    css_class = _nav_button_css_class(f"nav::{key_prefix}")
    style = "".join(e.proto.body for e in at.get("html"))
    match = re.search(rf"\.{re.escape(css_class)}\s*button\s*\{{[^}}]*padding-left:\s*(\d+)px", style)
    assert match, f"no indent rule found for {css_class!r} in:\n{style}"
    return int(match.group(1))


def _variance_values_by_label(widget) -> dict[str, float]:
    """label -> decision-variance-explained value read off any dropdown
    widget's own "(NN% variance explained)"-suffixed options (see
    _strip_variance_suffix/_decision_variance_by_key)."""
    values = {}
    for option in widget.options:
        match = re.match(r"^(.*)  \((\d+)% variance explained\)$", option)
        if match:
            label, value = match.groups()
            values[label] = float(value)
    return values


def _option_labels(widget) -> set[str]:
    """The plain feature labels a variance-suffixed multiselect widget
    currently offers (see _strip_variance_suffix) -- AppTest's own
    `.options` already reflects the widget's format_func output, not the
    raw feature-key identity."""
    return {_strip_variance_suffix(o) for o in widget.options}


def _variance_values_by_key(at: AppTest, widget_key: str, feature_keys=_FEATURE_KEYS) -> dict[str, float]:
    """Reads the "(NN% variance explained)" values off the multiselect at
    `widget_key` (e.g. f"{key_prefix}split_by" or
    f"{key_prefix}claim_filter_keys") -- computed over whatever pool that
    widget's own dropdown ranks against (see _render_substrategy), so it
    reflects whatever claim filters are currently narrowing that node."""
    label_to_key = {cfr_features.feature_label(k): k for k in feature_keys}
    values: dict[str, float] = {}
    for label, value in _variance_values_by_label(at.multiselect(key=widget_key)).items():
        if label in label_to_key:
            values[label_to_key[label]] = value
    return values


def _batch_set(at: AppTest, values: dict[str, list[str]]) -> AppTest:
    """Sets several multiselect widgets' values together before a single
    .run() -- AppTest's own widget-state round-trip for a format_func-based
    multiselect (every widget in this app that shows a "(NN% variance
    explained)" suffix) isn't reliably preserved for a widget that's *not*
    re-asserted in the same batch/run as some other widget's change
    (confirmed by inspecting raw st.session_state directly during
    development -- a testing-harness quirk, not real app behavior; a real
    browser session doesn't have this problem, since it tracks each
    option's raw identity client-side instead of round-tripping through
    format_func on every interaction). The safe pattern used throughout
    this file: batch every widget that's being given its *first* value
    together in one .run(), and keep any dependent widget that only exists
    *after* an earlier one has a value (e.g. "keep values" only appearing
    once a claim feature is picked) to its own preceding .run()."""
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
    """Clicks "Add sub-strategy" under `parent_prefix` (which must
    currently be the selected/rendered node, for its own button to exist)
    and returns the new child's own key_prefix -- which, since adding a
    sub-strategy also auto-selects it (see _add_substrategy in
    cfr_explorer.py), is now the *currently selected* node too, so its own
    widgets are immediately available without a separate _select call."""
    at.button(key=f"{parent_prefix}add_substrategy").click().run(timeout=60)
    return _substrategy_child_prefix(at, parent_prefix)


def _select(at: AppTest, key_prefix: str) -> AppTest:
    """Switches the central column to show `key_prefix`'s own page --
    equivalent to clicking that node's sidebar/quick-jump button (see
    _select_node in cfr_explorer.py), done here by setting session_state
    directly, matching how this file already reaches into
    at.session_state["substrategy_children"] elsewhere."""
    at.session_state["selected_substrategy"] = key_prefix
    at.run(timeout=60)
    return at


def _own_heading(at: AppTest) -> str:
    """The currently selected sub-strategy's own heading text -- every
    node's heading (root or not) is a plain st.header now (see
    _render_substrategy), and at.header also always picks up the
    sidebar's own "Navigation" header, so this filters that one out."""
    return next(h.value for h in at.header if h.value != "Navigation")


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
    # of recomputing the net's own action-probability predictions (the slow
    # part -- see _build_dataframe) from scratch for all 200+ samples on
    # every single test.
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


def _make_correlated_checkpoint(path: str, rng: np.random.Generator, num_samples: int = 4000) -> None:
    """Like _make_checkpoint, except hand_category_norm is pinned to
    exactly 0.0 whenever street_norm's own value falls in the "Preflop"
    bucket -- mirroring a real feature like num_overcards_norm, which is
    always exactly 0 preflop since there's no board yet to have overcards
    on. Filtering to Preflop should then make hand_category_norm's
    decision variance explained collapse to exactly 0%. `num_samples`
    defaults much higher than _make_checkpoint's own 200: hand_category_norm
    has 26 observed levels, and _decision_variance_explained's own
    leave-one-out correction (see its docstring) needs enough samples per
    level to tell a real, if diluted, relationship apart from noise --
    too few and the honest, cross-validated read is indistinguishable
    from 0% even for a real effect, which would make this checkpoint's
    own "meaningfully important unfiltered" precondition fail before the
    Preflop filter is even applied."""
    feature_dim = len(cfr_features.feature_indices(_FEATURE_KEYS))
    torch.manual_seed(0)  # AdvantageNet's init otherwise draws from torch's unseeded global RNG
    net = cfr_networks.AdvantageNet(input_dim=feature_dim, hidden_sizes=(8, 8))
    street_idx = _FEATURE_KEYS.index("street_norm")
    hand_idx = _FEATURE_KEYS.index("hand_category_norm")
    with torch.no_grad():
        net.hidden[0].block[0].weight[:, hand_idx] *= 20.0  # give it an outsized true effect over the whole reservoir
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


def _make_overfitting_checkpoint(path: str, rng: np.random.Generator, num_samples: int = 1000) -> None:
    """Regression coverage for a real bug in cfr_explorer.
    _decision_variance_explained (see TestSuggestedSubstrategyButtons.
    test_best_second_split_by_does_not_favor_a_rare_noise_candidate):
    before its leave-one-out correction, scoring a candidate second Split
    By feature by how well grouping *by it* fits this node's own claimed
    rows let a candidate that's overwhelmingly one value with just a
    handful of stray rows in another bucket (>= 2 observed categories, so
    not literally constant) outscore a genuinely informative candidate --
    those stray rows land in their own tiny/singleton (first, candidate)
    groups, which trivially "explain" their own members with zero error
    regardless of whether the feature is actually predictive of anything.

    hand_category_norm's own net input weight is zeroed out below, so the
    net's output is mathematically guaranteed not to depend on it at all
    -- yet it's still observed at 2 values here (995 rows at one, 5
    rng-placed stray rows at the other), exactly the shape that triggered
    the bug. Those 5 rows also get an extreme street_norm reading (0.999)
    each, so the net's own real (if more modest than hole_suited's)
    street_norm sensitivity gives them a genuinely elevated, outlier-sized
    residual -- shared with plenty of other ordinary high-street_norm rows
    under street_norm's own grouping (diluted across a real-sized group,
    same bucket), but isolated into hand_category_norm's own tiny
    rare-value groups instead, which then trivially "explain" them away
    almost entirely. Without that shared outlier magnitude, 5 rows out of
    1000 is too small a share to reliably reproduce the bug -- landing in
    an unremarkable spot wouldn't give a tiny group anything to trivially
    over-explain."""
    feature_dim = len(cfr_features.feature_indices(_FEATURE_KEYS))
    hole_idx = _FEATURE_KEYS.index("hole_suited")
    street_idx = _FEATURE_KEYS.index("street_norm")
    hand_idx = _FEATURE_KEYS.index("hand_category_norm")

    torch.manual_seed(0)  # AdvantageNet's init otherwise draws from torch's unseeded global RNG
    net = cfr_networks.AdvantageNet(input_dim=feature_dim, hidden_sizes=(8, 8))
    with torch.no_grad():
        net.hidden[0].block[0].weight[:, hole_idx] *= 20.0
        net.hidden[0].block[0].weight[:, street_idx] *= 8.0
        net.hidden[0].block[0].weight[:, hand_idx] = 0.0
    net_config = cfr_networks.AdvantageNetConfig(feature_keys=_FEATURE_KEYS, hidden_sizes=(8, 8), table_size=3)
    cfr_networks.save(net, net_config, path)

    reservoir = cfr_reservoir.ReservoirBuffer(
        capacity=num_samples, feature_dim=feature_dim, num_actions=strategy.NUM_ACTION_CATEGORIES, rng=rng,
    )
    rare_rows = set(rng.choice(num_samples, size=5, replace=False).tolist())
    for i in range(num_samples):
        features = rng.random(feature_dim).astype(np.float32)
        if i in rare_rows:
            features[hand_idx] = 1.0
            features[street_idx] = 0.999
        regrets = rng.normal(size=strategy.NUM_ACTION_CATEGORIES).astype(np.float32)
        legal_mask = rng.random(strategy.NUM_ACTION_CATEGORIES) > 0.3
        legal_mask[strategy.ACTION_CALL] = True
        reservoir.add(features, regrets, legal_mask, float(rng.integers(1, 10)))
    reservoir.save(path)


@pytest.fixture(scope="module")
def _overfitting_checkpoint_path(tmp_path_factory) -> str:
    path = os.path.join(str(tmp_path_factory.mktemp("cfr_explorer_overfitting_checkpoint")), "checkpoint")
    _make_overfitting_checkpoint(path, np.random.default_rng(0))
    return path


@pytest.fixture
def overfitting_checkpoint(_overfitting_checkpoint_path, monkeypatch):
    monkeypatch.setenv("CFR_EXPLORER_CHECKPOINT_PATH", _overfitting_checkpoint_path)
    return _overfitting_checkpoint_path


def _make_importance_per_level_checkpoint(path: str, rng: np.random.Generator, num_samples: int = 600) -> None:
    """Regression coverage for cfr_explorer._add_max_importance_split's
    per-level normalization (see TestSuggestedSubstrategyButtons.
    test_max_importance_split_prefers_higher_importance_per_level):
    hand_category_norm gets a bigger net input weight than street_norm --
    a higher *raw* importance -- but hand_category_norm has 26
    observed levels (see cfr_features.bucket_categories) against
    street_norm's 4, so street_norm's importance *per level* is actually
    higher. "Add maximum importance split" should add street_norm (fewer
    sub-strategies for a person to learn per unit of importance gained),
    not hand_category_norm (raw-importance winner, but 26 new
    sub-strategies for comparatively little additional payoff each).
    hole_suited's own net input weight is zeroed out entirely so it can't
    accidentally outscore either one."""
    feature_dim = len(cfr_features.feature_indices(_FEATURE_KEYS))
    hole_idx = _FEATURE_KEYS.index("hole_suited")
    street_idx = _FEATURE_KEYS.index("street_norm")
    hand_idx = _FEATURE_KEYS.index("hand_category_norm")

    torch.manual_seed(0)  # AdvantageNet's init otherwise draws from torch's unseeded global RNG
    net = cfr_networks.AdvantageNet(input_dim=feature_dim, hidden_sizes=(8, 8))
    with torch.no_grad():
        net.hidden[0].block[0].weight[:, hole_idx] = 0.0
        net.hidden[0].block[0].weight[:, street_idx] *= 15.0
        net.hidden[0].block[0].weight[:, hand_idx] *= 18.0
    net_config = cfr_networks.AdvantageNetConfig(feature_keys=_FEATURE_KEYS, hidden_sizes=(8, 8), table_size=3)
    cfr_networks.save(net, net_config, path)

    reservoir = cfr_reservoir.ReservoirBuffer(
        capacity=num_samples, feature_dim=feature_dim, num_actions=strategy.NUM_ACTION_CATEGORIES, rng=rng,
    )
    for _ in range(num_samples):
        features = rng.random(feature_dim).astype(np.float32)
        regrets = rng.normal(size=strategy.NUM_ACTION_CATEGORIES).astype(np.float32)
        legal_mask = rng.random(strategy.NUM_ACTION_CATEGORIES) > 0.3
        legal_mask[strategy.ACTION_CALL] = True
        reservoir.add(features, regrets, legal_mask, float(rng.integers(1, 10)))
    reservoir.save(path)


@pytest.fixture(scope="module")
def _importance_per_level_checkpoint_path(tmp_path_factory) -> str:
    path = os.path.join(str(tmp_path_factory.mktemp("cfr_explorer_importance_per_level_checkpoint")), "checkpoint")
    _make_importance_per_level_checkpoint(path, np.random.default_rng(0))
    return path


@pytest.fixture
def importance_per_level_checkpoint(_importance_per_level_checkpoint_path, monkeypatch):
    monkeypatch.setenv("CFR_EXPLORER_CHECKPOINT_PATH", _importance_per_level_checkpoint_path)
    return _importance_per_level_checkpoint_path


def _make_interaction_per_level_checkpoint(path: str, n: int = 8000, seed: int = 0) -> None:
    """Regression coverage for cfr_explorer._add_max_interaction_split's
    per-level normalization (see TestSuggestedSubstrategyButtons.
    test_max_interaction_split_prefers_higher_interaction_per_level):
    a trained net whose output genuinely, causally interacts with
    hole_suited via *both* street_norm and hand_category_norm -- sign-
    reversing hole_suited's own effect on one street bucket (Flop, 1 of
    street_norm's 4) and, more broadly, on hand_category_norm readings
    below 0.5 (roughly half of its 26 observed levels) -- but with a
    bigger reversal magnitude for the hand_category_norm interaction, so
    its *raw* (summed/averaged, not per-level) interaction strength with
    hole_suited comes out higher than street_norm's, the same way
    _make_importance_per_level_checkpoint engineers a raw-importance
    winner with many levels. street_norm's own interaction, spread over
    only 4 levels instead of 26, is higher *per level* -- "Add maximum
    interaction split" should add street_norm, not hand_category_norm."""
    feature_dim = len(cfr_features.feature_indices(_FEATURE_KEYS))
    suited_idx = _FEATURE_KEYS.index("hole_suited")
    street_idx = _FEATURE_KEYS.index("street_norm")
    hand_idx = _FEATURE_KEYS.index("hand_category_norm")
    rng = np.random.default_rng(seed)
    suited = (rng.random(n) < 0.5).astype(np.float32)
    street_choice = rng.integers(0, 4, size=n)
    street = np.array([0.0, 1 / 3, 2 / 3, 1.0], dtype=np.float32)[street_choice]
    is_flop = street_choice == 1
    hand_raw = rng.random(n).astype(np.float32)
    is_weak_hand = hand_raw < 0.5

    signal_street = np.where(is_flop, -20.0 * suited, 20.0 * suited)
    signal_hand = np.where(is_weak_hand, -22.0 * suited, 22.0 * suited)

    X = np.zeros((n, feature_dim), dtype=np.float32)
    X[:, suited_idx] = suited
    X[:, street_idx] = street
    X[:, hand_idx] = hand_raw
    y = np.zeros((n, strategy.NUM_ACTION_CATEGORIES), dtype=np.float32)
    y[:, 0] = signal_street + rng.normal(scale=0.1, size=n)  # isolate each signal to its own action
    y[:, 1] = signal_hand + rng.normal(scale=0.1, size=n)
    y[:, 2:] = rng.normal(scale=0.1, size=(n, strategy.NUM_ACTION_CATEGORIES - 2))

    torch.manual_seed(0)
    net = cfr_networks.AdvantageNet(input_dim=feature_dim, hidden_sizes=(32, 32), dropout=0.0)
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    net.train()
    train_rng = np.random.default_rng(0)
    for _ in range(2000):
        idx = train_rng.integers(0, n, size=256)
        pred = net(Xt[idx])
        loss = ((pred - yt[idx]) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    net.eval()

    net_config = cfr_networks.AdvantageNetConfig(feature_keys=_FEATURE_KEYS, hidden_sizes=(32, 32), table_size=3)
    cfr_networks.save(net, net_config, path)

    reservoir = cfr_reservoir.ReservoirBuffer(
        capacity=n, feature_dim=feature_dim, num_actions=strategy.NUM_ACTION_CATEGORIES, rng=rng,
    )
    for i in range(n):
        regrets = rng.normal(size=strategy.NUM_ACTION_CATEGORIES).astype(np.float32)
        legal_mask = rng.random(strategy.NUM_ACTION_CATEGORIES) > 0.3
        legal_mask[strategy.ACTION_CALL] = True
        reservoir.add(X[i], regrets, legal_mask, float(rng.integers(1, 10)))
    reservoir.save(path)


@pytest.fixture(scope="module")
def _interaction_per_level_checkpoint_path(tmp_path_factory) -> str:
    path = os.path.join(str(tmp_path_factory.mktemp("cfr_explorer_interaction_per_level_checkpoint")), "checkpoint")
    _make_interaction_per_level_checkpoint(path)
    return path


@pytest.fixture
def interaction_per_level_checkpoint(_interaction_per_level_checkpoint_path, monkeypatch):
    monkeypatch.setenv("CFR_EXPLORER_CHECKPOINT_PATH", _interaction_per_level_checkpoint_path)
    return _interaction_per_level_checkpoint_path


_MARGINAL_INTERACTION_FEATURE_KEYS = ("street_norm", "hand_category_norm", "is_aggressor_previous_street")


def _make_marginal_interaction_checkpoint(path: str, n: int = 8000, seed: int = 0) -> None:
    """Regression coverage for _add_max_interaction_split's own
    per-level normalization dividing the *marginal* gain over the current
    Split By, not the joint total (see its own docstring): street_norm
    (this checkpoint's own current Split By, 4 levels) genuinely explains
    action column 0 entirely on its own; hand_category_norm (26 levels)
    genuinely explains action column 1, which street_norm alone can't
    touch at all; is_aggressor_previous_street (2 levels) explains nothing anywhere -- a
    pure no-op feature. So street_norm alone already carries a real,
    substantial baseline, hand_category_norm's own *marginal* gain over
    that baseline is large, and is_aggressor_previous_street's marginal gain is exactly
    zero. Dividing the old way (joint *total*, which includes street_norm's
    own already-explained-anyway baseline) by level count made is_aggressor_previous_street
    (baseline/2, a big number over a tiny divisor) beat hand_category_norm
    (baseline+real gain, over 26) despite contributing nothing new --
    "Add maximum interaction split" should pick hand_category_norm."""
    feature_dim = len(cfr_features.feature_indices(_MARGINAL_INTERACTION_FEATURE_KEYS))
    street_idx = _MARGINAL_INTERACTION_FEATURE_KEYS.index("street_norm")
    hand_idx = _MARGINAL_INTERACTION_FEATURE_KEYS.index("hand_category_norm")
    aggressor_idx = _MARGINAL_INTERACTION_FEATURE_KEYS.index("is_aggressor_previous_street")
    rng = np.random.default_rng(seed)

    street_choice = rng.integers(0, 4, size=n)
    street = np.array([0.0, 1 / 3, 2 / 3, 1.0], dtype=np.float32)[street_choice]
    is_flop = street_choice == 1
    hand_raw = rng.random(n).astype(np.float32)
    is_weak_hand = hand_raw < 0.5
    aggressor = rng.random(n).astype(np.float32)

    signal_street = np.where(is_flop, -20.0, 20.0)
    signal_hand = np.where(is_weak_hand, -20.0, 20.0)

    X = np.zeros((n, feature_dim), dtype=np.float32)
    X[:, street_idx] = street
    X[:, hand_idx] = hand_raw
    X[:, aggressor_idx] = aggressor
    y = np.zeros((n, strategy.NUM_ACTION_CATEGORIES), dtype=np.float32)
    y[:, 0] = signal_street + rng.normal(scale=0.1, size=n)  # fully explained by street_norm alone
    y[:, 1] = signal_hand + rng.normal(scale=0.1, size=n)  # street_norm alone can't touch this
    y[:, 2:] = rng.normal(scale=0.1, size=(n, strategy.NUM_ACTION_CATEGORIES - 2))

    torch.manual_seed(0)
    net = cfr_networks.AdvantageNet(input_dim=feature_dim, hidden_sizes=(32, 32), dropout=0.0)
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    net.train()
    train_rng = np.random.default_rng(0)
    for _ in range(2000):
        idx = train_rng.integers(0, n, size=256)
        pred = net(Xt[idx])
        loss = ((pred - yt[idx]) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    net.eval()

    net_config = cfr_networks.AdvantageNetConfig(
        feature_keys=_MARGINAL_INTERACTION_FEATURE_KEYS, hidden_sizes=(32, 32), table_size=3,
    )
    cfr_networks.save(net, net_config, path)

    reservoir = cfr_reservoir.ReservoirBuffer(
        capacity=n, feature_dim=feature_dim, num_actions=strategy.NUM_ACTION_CATEGORIES, rng=rng,
    )
    for i in range(n):
        regrets = rng.normal(size=strategy.NUM_ACTION_CATEGORIES).astype(np.float32)
        legal_mask = rng.random(strategy.NUM_ACTION_CATEGORIES) > 0.3
        legal_mask[strategy.ACTION_CALL] = True
        reservoir.add(X[i], regrets, legal_mask, float(rng.integers(1, 10)))
    reservoir.save(path)


@pytest.fixture(scope="module")
def _marginal_interaction_checkpoint_path(tmp_path_factory) -> str:
    path = os.path.join(str(tmp_path_factory.mktemp("cfr_explorer_marginal_interaction_checkpoint")), "checkpoint")
    _make_marginal_interaction_checkpoint(path)
    return path


@pytest.fixture
def marginal_interaction_checkpoint(_marginal_interaction_checkpoint_path, monkeypatch):
    monkeypatch.setenv("CFR_EXPLORER_CHECKPOINT_PATH", _marginal_interaction_checkpoint_path)
    return _marginal_interaction_checkpoint_path


_THREE_WAY_FEATURE_KEYS = ("street_norm", "hole_suited", "connected_flop", "is_aggressor_previous_street")


def _make_three_way_checkpoint(path: str, rng: np.random.Generator, num_samples: int = 2000) -> None:
    """Regression coverage for cfr_explorer._group_labels_for_rows'
    general (non-Exact-Hole-Hand) branch, which used to combine only its
    *first two* group_by_keys and silently drop any beyond that -- fine as
    long as every caller passed at most 2 keys (Split By itself always
    has), but _decision_variance_by_key's own `pair_with` can legitimately
    hold 2 keys already (this node's own full, 2-feature Split By) plus
    one more candidate being scored against it -- 3 keys total. With the
    3rd silently dropped, every remaining candidate's "interaction with
    current Split By" collapsed to re-measuring the *same* 2-key grouping
    regardless of which candidate was asked about, so every row in the
    feature table's own Interaction column read identically once 2 Split
    By features were already chosen (see TestFeatureTable.
    test_interaction_column_differs_per_candidate_with_two_split_by_features).

    street_norm and hole_suited get real net input weight (the "already
    chosen" Split By pair); connected_flop also gets real weight (a
    genuinely informative remaining candidate); is_aggressor_previous_street's own weight
    is zeroed out (an uninformative one). Both remaining candidates are
    plain booleans (2 observed levels each, like street_norm/hole_suited
    themselves) specifically so their own 3-way joint partition ends up
    the same *size* regardless of which one is asked about -- isolating
    the bug (a structurally wrong, identical computation) from the
    leave-one-out correction's own separate, expected sensitivity to
    group cardinality (see _decision_variance_explained)."""
    feature_dim = len(cfr_features.feature_indices(_THREE_WAY_FEATURE_KEYS))
    street_idx = _THREE_WAY_FEATURE_KEYS.index("street_norm")
    hole_idx = _THREE_WAY_FEATURE_KEYS.index("hole_suited")
    connected_idx = _THREE_WAY_FEATURE_KEYS.index("connected_flop")
    aggressor_idx = _THREE_WAY_FEATURE_KEYS.index("is_aggressor_previous_street")

    torch.manual_seed(0)  # AdvantageNet's init otherwise draws from torch's unseeded global RNG
    net = cfr_networks.AdvantageNet(input_dim=feature_dim, hidden_sizes=(8, 8))
    with torch.no_grad():
        net.hidden[0].block[0].weight[:, street_idx] *= 12.0
        net.hidden[0].block[0].weight[:, hole_idx] *= 12.0
        net.hidden[0].block[0].weight[:, connected_idx] *= 15.0
        net.hidden[0].block[0].weight[:, aggressor_idx] = 0.0
    net_config = cfr_networks.AdvantageNetConfig(
        feature_keys=_THREE_WAY_FEATURE_KEYS, hidden_sizes=(8, 8), table_size=3,
    )
    cfr_networks.save(net, net_config, path)

    reservoir = cfr_reservoir.ReservoirBuffer(
        capacity=num_samples, feature_dim=feature_dim, num_actions=strategy.NUM_ACTION_CATEGORIES, rng=rng,
    )
    for _ in range(num_samples):
        features = rng.random(feature_dim).astype(np.float32)
        regrets = rng.normal(size=strategy.NUM_ACTION_CATEGORIES).astype(np.float32)
        legal_mask = rng.random(strategy.NUM_ACTION_CATEGORIES) > 0.3
        legal_mask[strategy.ACTION_CALL] = True
        reservoir.add(features, regrets, legal_mask, float(rng.integers(1, 10)))
    reservoir.save(path)


@pytest.fixture(scope="module")
def _three_way_checkpoint_path(tmp_path_factory) -> str:
    path = os.path.join(str(tmp_path_factory.mktemp("cfr_explorer_three_way_checkpoint")), "checkpoint")
    _make_three_way_checkpoint(path, np.random.default_rng(0))
    return path


@pytest.fixture
def three_way_checkpoint(_three_way_checkpoint_path, monkeypatch):
    monkeypatch.setenv("CFR_EXPLORER_CHECKPOINT_PATH", _three_way_checkpoint_path)
    return _three_way_checkpoint_path


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


class TestSubstrategyImportance:
    def test_narrowing_a_filter_changes_reported_importance_values(self, correlated_checkpoint):
        # synthetic_checkpoint's own net is untrained/random with no real
        # structure, so under the leave-one-out-corrected metric (unlike the
        # old raw-SHAP one) it honestly reads 0% everywhere regardless of
        # filtering -- there's nothing there to detect. correlated_checkpoint
        # has a real, engineered relationship (see _make_correlated_checkpoint)
        # for filtering to actually change.
        at = _run_app()
        before = _variance_values_by_key(at, "root::split_by")

        at.multiselect(key="root::claim_filter_keys").set_value(["street_norm"]).run(timeout=60)
        _batch_set(at, {"root::claim_filter_values::street_norm": ["Preflop"]})

        after = _variance_values_by_key(at, "root::split_by")
        assert not at.exception
        assert before != after

    def test_feature_pinned_constant_by_the_filter_scores_exactly_zero(self, correlated_checkpoint):
        # Regression test: hand_category_norm is pinned to a single constant
        # value whenever street_norm falls in the Preflop bucket (see
        # _make_correlated_checkpoint). Grouping by a constant feature (see
        # _decision_variance_explained) produces exactly the same partition
        # as no grouping at all, so it explains exactly 0% -- not just a
        # small amount -- once filtered to Preflop.
        at = _run_app()
        unfiltered = _variance_values_by_key(at, "root::split_by")

        at.multiselect(key="root::claim_filter_keys").set_value(["street_norm"]).run(timeout=60)
        _batch_set(at, {"root::claim_filter_values::street_norm": ["Preflop"]})

        assert not at.exception
        filtered = _variance_values_by_key(at, "root::split_by")
        # meaningfully important over the whole reservoir, so the drop to
        # exactly 0% below is a real change, not just noise around a value
        # that was already near 0%.
        assert unfiltered["hand_category_norm"] > 1.0
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
        labels = {_strip_variance_suffix(o) for o in cross.options}
        assert "Betting Street" not in labels  # street_norm's own label -- can't cross a feature with itself
        assert labels == {"Suited Hole Cards", "Hand Strength Tier"}

    def test_cross_dropdown_options_are_labeled_with_interaction_strength_and_ranked_by_it(self, synthetic_checkpoint):
        at = _run_app()
        at.multiselect(key="root::extra_graph").set_value(["street_norm"]).run(timeout=60)

        cross = at.multiselect(key="root::extra::graph_cross::street_norm")
        values = _variance_values_by_label(cross)
        assert set(values) == {"Suited Hole Cards", "Hand Strength Tier"}
        assert all(v >= 0.0 for v in values.values())
        # Options themselves (not just the parsed values) come back
        # strongest-interaction-first.
        ordered_values = [values[_strip_variance_suffix(o)] for o in cross.options]
        assert ordered_values == sorted(ordered_values, reverse=True)

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

    def test_graphs_heading_renders_as_a_subheader_at_root(self, synthetic_checkpoint):
        at = _run_app()
        at.multiselect(key="root::extra_graph").set_value(["hole_suited"]).run(timeout=60)

        assert not at.exception
        assert any(h.value == "Graphs" for h in at.subheader)

    def test_graphs_heading_renders_as_a_subheader_for_a_substrategy(self, synthetic_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")
        at.multiselect(key=f"{c0}extra_graph").set_value(["hole_suited"]).run(timeout=60)

        assert not at.exception
        assert any(h.value == "Graphs" for h in at.subheader)

    def test_each_substrategy_gets_its_own_independently_keyed_graph(self, synthetic_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")  # auto-selected
        at.multiselect(key=f"{c0}extra_graph").set_value(["hole_suited"]).run(timeout=60)
        assert not at.exception
        c0_charts = at.get("plotly_chart")
        assert len(c0_charts) == 1
        assert c0_charts[0].id.endswith(f"{c0}extra::graph_chart::hole_suited")

        _select(at, "root::")
        at.multiselect(key="root::extra_graph").set_value(["hole_suited"]).run(timeout=60)
        assert not at.exception
        root_charts = at.get("plotly_chart")
        assert len(root_charts) == 1
        assert root_charts[0].id.endswith("root::extra::graph_chart::hole_suited")

        assert root_charts[0].id != c0_charts[0].id

    def test_substrategy_cross_dropdown_is_independent_of_roots_own(self, synthetic_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")  # auto-selected

        _select(at, "root::")
        at.multiselect(key="root::extra_graph").set_value(["hole_suited"]).run(timeout=60)
        at.multiselect(key="root::extra::graph_cross::hole_suited").set_value(["hand_category_norm"])
        at.run(timeout=60)
        assert not at.exception

        _select(at, c0)
        at.multiselect(key=f"{c0}extra_graph").set_value(["hole_suited"]).run(timeout=60)
        assert not at.exception
        substrategy_cross = at.multiselect(key=f"{c0}extra::graph_cross::hole_suited")
        assert substrategy_cross.value == []  # unaffected by root's own earlier selection


class TestSuggestedSubstrategyButtons:
    """The three "Suggested sub-strategies" buttons (see
    _render_substrategy/_add_max_interaction_split/_add_max_importance_split/
    _add_best_second_split_by) -- algorithmic shortcuts for the same claim-
    filter/Split By state a person could otherwise only set up by hand."""

    def test_max_interaction_split_disabled_without_a_split_by(self, synthetic_checkpoint):
        at = _run_app()
        assert at.button(key="root::add_max_interaction_split").proto.disabled

    def test_max_interaction_split_enabled_once_a_split_by_is_chosen(self, synthetic_checkpoint):
        at = _run_app()
        at.multiselect(key="root::split_by").set_value(["street_norm"]).run(timeout=60)
        assert not at.button(key="root::add_max_interaction_split").proto.disabled

    def test_max_interaction_split_adds_one_child_per_observed_level_of_one_feature(self, synthetic_checkpoint):
        at = _run_app()
        at.multiselect(key="root::split_by").set_value(["street_norm"]).run(timeout=60)
        at.button(key="root::add_max_interaction_split").click().run(timeout=60)

        assert not at.exception
        children = at.session_state["substrategy_children"]["root::"]
        assert len(children) >= 2  # at least 2 levels for a real split to make sense

        claims = at.session_state["substrategy_claims"]
        split_bys = at.session_state["substrategy_split_by"]
        for child_id in children:
            child_prefix = f"root::substrategy_{child_id}::"
            claim = claims[child_prefix]
            assert set(claim.keys()) != {"street_norm"}  # never re-splits on its own Split By feature
            assert len(claim) == 1  # one feature, exactly one kept level
            assert len(next(iter(claim.values()))) == 1
            assert split_bys[child_prefix] == ["street_norm"]  # inherits the parent's own current Split By

        # Every child claims a different level of the same one feature.
        claimed_features = {next(iter(claims[f"root::substrategy_{cid}::"])) for cid in children}
        assert len(claimed_features) == 1

    def test_max_interaction_split_prefers_higher_interaction_per_level(self, interaction_per_level_checkpoint):
        # Regression coverage: hand_category_norm has the higher *raw*
        # interaction strength with hole_suited in this checkpoint (see
        # _make_interaction_per_level_checkpoint), but it spreads that
        # interaction across 26 observed levels against street_norm's 4, so
        # street_norm's interaction *per level* is actually higher -- "Add
        # maximum interaction split" should pick street_norm (fewer new
        # sub-strategies to learn per unit of interaction gained), not
        # hand_category_norm (the raw-interaction winner).
        at = _run_app()
        at.multiselect(key="root::split_by").set_value(["hole_suited"]).run(timeout=60)
        at.button(key="root::add_max_interaction_split").click().run(timeout=60)

        assert not at.exception
        children = at.session_state["substrategy_children"]["root::"]
        claims = at.session_state["substrategy_claims"]
        claimed_features = {next(iter(claims[f"root::substrategy_{cid}::"])) for cid in children}
        assert claimed_features == {"street_norm"}

    def test_max_interaction_split_normalizes_the_marginal_gain_not_the_joint_total(
        self, marginal_interaction_checkpoint,
    ):
        # Regression coverage: is_aggressor_previous_street contributes nothing beyond
        # what street_norm (this node's own current Split By) already
        # explains alone, while hand_category_norm genuinely explains a
        # large share street_norm alone can't touch (see
        # _make_marginal_interaction_checkpoint). Dividing the joint
        # *total* (which always also includes street_norm's own
        # already-explained-anyway baseline) by level count let
        # is_aggressor_previous_street's 2 levels beat hand_category_norm's 26 purely
        # because that shared baseline is a bigger number than
        # hand_category_norm's own genuine marginal gain -- dividing the
        # marginal gain instead should pick hand_category_norm.
        at = _run_app()
        at.multiselect(key="root::split_by").set_value(["street_norm"]).run(timeout=60)
        at.button(key="root::add_max_interaction_split").click().run(timeout=60)

        assert not at.exception
        children = at.session_state["substrategy_children"]["root::"]
        claims = at.session_state["substrategy_claims"]
        claimed_features = {next(iter(claims[f"root::substrategy_{cid}::"])) for cid in children}
        assert claimed_features == {"hand_category_norm"}

    def test_max_importance_split_works_without_any_split_by_chosen(self, synthetic_checkpoint):
        at = _run_app()
        assert not at.button(key="root::add_max_importance_split").proto.disabled
        at.button(key="root::add_max_importance_split").click().run(timeout=60)

        assert not at.exception
        assert len(at.session_state["substrategy_children"]["root::"]) >= 2

    def test_max_importance_split_prefers_higher_importance_per_level(self, importance_per_level_checkpoint):
        # Regression coverage: hand_category_norm has the higher *raw*
        # importance in this checkpoint (see
        # _make_importance_per_level_checkpoint), but it spreads that
        # importance across 26 observed levels against street_norm's 4, so
        # street_norm's importance *per level* is actually higher --
        # "Add maximum importance split" should pick street_norm (fewer new
        # sub-strategies to learn per unit of importance), not
        # hand_category_norm (the raw-importance winner).
        at = _run_app()
        at.button(key="root::add_max_importance_split").click().run(timeout=60)

        assert not at.exception
        children = at.session_state["substrategy_children"]["root::"]
        claims = at.session_state["substrategy_claims"]
        claimed_features = {next(iter(claims[f"root::substrategy_{cid}::"])) for cid in children}
        assert claimed_features == {"street_norm"}

    def test_best_second_split_by_disabled_unless_exactly_one_chosen(self, synthetic_checkpoint):
        at = _run_app()
        assert at.button(key="root::add_best_second_split_by").proto.disabled  # 0 chosen

        at.multiselect(key="root::split_by").set_value(["street_norm", "hole_suited"]).run(timeout=60)
        assert at.button(key="root::add_best_second_split_by").proto.disabled  # 2 chosen

    def test_best_second_split_by_adds_a_second_feature(self, synthetic_checkpoint):
        at = _run_app()
        at.multiselect(key="root::split_by").set_value(["street_norm"]).run(timeout=60)
        assert not at.button(key="root::add_best_second_split_by").proto.disabled

        at.button(key="root::add_best_second_split_by").click().run(timeout=60)

        assert not at.exception
        value = at.multiselect(key="root::split_by").value
        assert len(value) == 2
        assert "street_norm" in value

    def test_best_second_split_by_respects_current_filters(self, correlated_checkpoint):
        # Regression coverage for a real bug: _add_best_second_split_by
        # scored every other configured feature as a candidate, unlike
        # _add_max_interaction_split/_add_max_importance_split (both go
        # through _splittable_candidates, which excludes anything constant
        # within this node's own claimed default_df). Filtering this child
        # to Preflop makes both remaining features constant there --
        # street_norm by the claim filter itself, hand_category_norm by the
        # checkpoint's own correlation (see _make_correlated_checkpoint) --
        # so there's nothing eligible left to pair with hole_suited, and
        # the click should be a no-op rather than picking one anyway.
        at = _run_app()
        at.multiselect(key="root::split_by").set_value(["hole_suited"]).run(timeout=60)
        c0 = _add_substrategy(at, "root::")
        at.multiselect(key=f"{c0}claim_filter_keys").set_value(["street_norm"]).run(timeout=60)
        _batch_set(at, {f"{c0}claim_filter_values::street_norm": ["Preflop"]})

        assert at.multiselect(key=f"{c0}split_by").value == ["hole_suited"]  # inherited
        assert not at.button(key=f"{c0}add_best_second_split_by").proto.disabled
        at.button(key=f"{c0}add_best_second_split_by").click().run(timeout=60)

        assert not at.exception
        assert at.multiselect(key=f"{c0}split_by").value == ["hole_suited"]

    def test_best_second_split_by_does_not_favor_a_rare_noise_candidate(self, overfitting_checkpoint):
        # Regression coverage for the deeper bug behind the filter-respecting
        # fix above: _decision_variance_explained itself could score a
        # candidate that's overwhelmingly one value with a handful of stray
        # rows in another bucket *higher* than a genuinely informative
        # candidate, purely because those stray rows land in their own
        # tiny/singleton groups and trivially "explain" themselves (see
        # _make_overfitting_checkpoint). hand_category_norm's own net input
        # weight is zeroed out in this checkpoint -- it cannot carry any
        # real signal -- while street_norm has a real (if more modest than
        # hole_suited's) one, so street_norm should legitimately win.
        at = _run_app()
        at.multiselect(key="root::split_by").set_value(["hole_suited"]).run(timeout=60)
        at.button(key="root::add_best_second_split_by").click().run(timeout=60)

        assert not at.exception
        assert at.multiselect(key="root::split_by").value == ["hole_suited", "street_norm"]

    def test_best_second_split_by_and_max_interaction_split_run_safely_under_exact_hole_hand(
        self, hole_hand_grid_checkpoint,
    ):
        # Regression coverage: Exact Hole Hand has no ordinary bucket-label
        # column (see _build_dataframe) -- picking it as this node's *sole*
        # current Split By pick (a valid, length-1 resolved_split_by) and
        # then asking either button to pair something else with it used to
        # raise a KeyError, before _group_labels_for_rows learned to group
        # it jointly with an ordinary feature's own bucket label (its grid
        # cell alongside the other feature's own value). Both buttons
        # should run without crashing -- "Add best second" ends up
        # collapsing back to Exact Hole Hand alone regardless of what it
        # picks (see _resolve_split_by), but "Add max interaction split"
        # genuinely uses the joint (grid cell, street_norm) grouping to
        # pick street_norm and split on it.
        at = _run_app()
        at.multiselect(key="root::split_by").set_value(["hole_hand_grid_x_norm"]).run(timeout=60)
        assert at.multiselect(key="root::split_by").value == ["hole_hand_grid_x_norm"]

        at.button(key="root::add_best_second_split_by").click().run(timeout=60)
        assert not at.exception

        at.button(key="root::add_max_interaction_split").click().run(timeout=60)
        assert not at.exception

    def test_optimise_split_by_picks_the_best_pair_regardless_of_current_choice(self, overfitting_checkpoint):
        # Regression/behavior coverage: unlike "Add best second Split By
        # feature", this button ignores whatever Split By is currently set
        # and searches every pair from scratch. Seeding Split By with
        # hand_category_norm -- not part of the true best pair in this
        # checkpoint (see _make_overfitting_checkpoint: its own net input
        # weight is zeroed out, hole_suited and street_norm both carry real
        # signal) -- proves it's a fresh search, not just an extension of
        # whatever was already chosen.
        at = _run_app()
        at.multiselect(key="root::split_by").set_value(["hand_category_norm"]).run(timeout=60)
        at.button(key="root::optimise_split_by").click().run(timeout=60)

        assert not at.exception
        assert set(at.multiselect(key="root::split_by").value) == {"hole_suited", "street_norm"}

    def test_optimise_split_by_no_ops_with_fewer_than_two_candidates(self, correlated_checkpoint):
        # Filtered to Preflop, both street_norm (by the filter itself) and
        # hand_category_norm (by the checkpoint's own correlation -- see
        # _make_correlated_checkpoint) are constant within this node's own
        # claimed rows, leaving only hole_suited -- one candidate, not
        # enough to form a pair.
        at = _run_app()
        c0 = _add_substrategy(at, "root::")
        at.multiselect(key=f"{c0}claim_filter_keys").set_value(["street_norm"]).run(timeout=60)
        _batch_set(at, {f"{c0}claim_filter_values::street_norm": ["Preflop"]})

        assert at.multiselect(key=f"{c0}split_by").value == []  # inherited from root's own default
        at.button(key=f"{c0}optimise_split_by").click().run(timeout=60)

        assert not at.exception
        assert at.multiselect(key=f"{c0}split_by").value == []


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
        assert "no filter set yet" in _own_heading(at)
        assert at.multiselect(key=f"{c0}claim_filter_keys")
        assert at.multiselect(key=f"{c0}split_by")
        assert at.button(key=f"{c0}add_substrategy")
        assert at.button(key=f"{c0}remove")
        assert at.button(key=f"{c0}move_up")
        assert at.button(key=f"{c0}move_down")
        assert at.button(key=f"{c0}back")

    def test_unclaimed_substrategy_sees_every_row_its_parent_has(self, synthetic_checkpoint):
        at = _run_app()
        _add_substrategy(at, "root::")
        assert not at.exception
        assert "(n=200)" in _own_heading(at)

    def test_parent_heading_shows_both_its_own_and_default_behaviour_counts(self, synthetic_checkpoint):
        # Regression coverage: a parent's own heading count (node_df, its
        # rule's total claimed scope) and its "Default behaviour" section
        # below (default_df, what's left once every child's own claim is
        # subtracted -- see _resolve_default_df) are two different numbers
        # by design, easily conflated since only the heading's own count
        # is shown right at the top. Once a child actually claims some
        # rows, the heading should show both, not just node_df's own
        # (unchanged, since node_df is deliberately unaffected by whatever
        # children go on to claim) count.
        at = _run_app()
        assert "(n=200)" in _own_heading(at)  # no children yet -- nothing to disambiguate

        c0 = _add_substrategy(at, "root::")
        at.multiselect(key=f"{c0}claim_filter_keys").set_value(["street_norm"]).run(timeout=60)
        observed = at.multiselect(key=f"{c0}claim_filter_values::street_norm").options
        _batch_set(at, {f"{c0}claim_filter_values::street_norm": observed[:1]})
        claimed_n = int(_own_heading(at).split("(n=")[1].split(")")[0])
        assert 0 < claimed_n < 200

        at.button(key=f"{c0}back").click().run(timeout=60)
        assert not at.exception
        header = _own_heading(at)
        assert f"(n=200, {200 - claimed_n} default)" in header

    def test_claim_filter_narrows_the_heading_count_and_names_the_feature(self, synthetic_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")
        at.multiselect(key=f"{c0}claim_filter_keys").set_value(["street_norm"]).run(timeout=60)
        observed = at.multiselect(key=f"{c0}claim_filter_values::street_norm").value
        _batch_set(at, {f"{c0}claim_filter_values::street_norm": observed[:1]})

        assert not at.exception
        assert observed[0] in _own_heading(at)
        assert "no filter set yet" not in _own_heading(at)

    def test_second_sibling_only_sees_rows_the_first_did_not_claim(self, synthetic_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")
        at.multiselect(key=f"{c0}claim_filter_keys").set_value(["street_norm"]).run(timeout=60)
        _batch_set(at, {f"{c0}claim_filter_values::street_norm": ["Preflop"]})
        c0_n = int(re.search(r"n=([\d,]+)", _own_heading(at)).group(1).replace(",", ""))

        _select(at, "root::")
        c1 = _add_substrategy(at, "root::")
        at.multiselect(key=f"{c1}claim_filter_keys").set_value(["street_norm"]).run(timeout=60)
        _batch_set(at, {f"{c1}claim_filter_values::street_norm": ["Flop"]})
        c1_n = int(re.search(r"n=([\d,]+)", _own_heading(at)).group(1).replace(",", ""))

        assert not at.exception
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
        _select(at, "root::")
        _add_substrategy(at, "root::")
        assert len(at.session_state["substrategy_children"]["root::"]) == 2

        _select(at, c0)
        at.button(key=f"{c0}remove").click().run(timeout=60)

        assert not at.exception
        assert len(at.session_state["substrategy_children"]["root::"]) == 1
        assert not any(b.key == f"{c0}add_substrategy" for b in at.button)

    def test_move_up_swaps_priority_order(self, synthetic_checkpoint):
        at = _run_app()
        _add_substrategy(at, "root::")
        _select(at, "root::")
        _add_substrategy(at, "root::")  # 2nd child now selected
        ids_before = list(at.session_state["substrategy_children"]["root::"])

        second_prefix = f"root::substrategy_{ids_before[1]}::"
        at.button(key=f"{second_prefix}move_up").click().run(timeout=60)

        assert not at.exception
        ids_after = list(at.session_state["substrategy_children"]["root::"])
        assert ids_after == [ids_before[1], ids_before[0]]

    def test_move_down_swaps_priority_order(self, synthetic_checkpoint):
        at = _run_app()
        _add_substrategy(at, "root::")
        _select(at, "root::")
        _add_substrategy(at, "root::")
        ids_before = list(at.session_state["substrategy_children"]["root::"])

        first_prefix = f"root::substrategy_{ids_before[0]}::"
        _select(at, first_prefix)
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
        assert "no filter set yet" in _own_heading(at)
        # "Back to parent" from the grandchild goes to c0, not root directly
        # -- confirming it really is nested one level under c0, not a
        # second direct child of root.
        at.button(key=f"{grandchild}back").click().run(timeout=60)
        assert not at.exception
        assert at.session_state["selected_substrategy"] == c0


def _max_eval_samples_input(at: AppTest):
    return at.number_input(key="max_eval_samples")


class TestMaxEvalSamples:
    """"Max samples to evaluate in any given spot" (see _capped_for_eval)
    -- decoupled from "Max reservoir samples to load" so a big loaded pool
    (needed to give an infrequent spot enough of its own matching rows for
    a meaningful Decision variance explained figure -- see
    _decision_variance_explained's own leave-one-out correction) doesn't
    also force every *other*, more common spot's own analysis to run over
    however many rows it happens to match."""

    def test_sidebar_control_exists_with_a_sane_default(self, synthetic_checkpoint):
        # Not hardcoding the exact default (DEFAULT_MAX_EVAL_SAMPLES) here
        # -- cfr_explorer.py isn't importable as a plain module (it calls
        # main() at module scope), and the exact value is expected to keep
        # being tuned -- just that the control exists, has no upper bound
        # (see the "no max input value" requirement), and defaults to
        # something positive.
        at = _run_app()
        control = _max_eval_samples_input(at)
        assert control.min == 100
        assert control.max is None
        assert control.value > 0

    def test_no_evaluated_on_note_when_under_the_cap(self, synthetic_checkpoint):
        at = _run_app()
        assert "evaluated on" not in _own_heading(at)

    def test_evaluated_on_note_appears_once_claimed_rows_exceed_the_cap(self, synthetic_checkpoint):
        # synthetic_checkpoint has 200 rows -- lowering the cap below that
        # (rather than building a dedicated 10,000+-row checkpoint just for
        # this) is the cheap way to exercise the "over the cap" path. 150,
        # not some smaller round number, since the control's own
        # min_value is 100.
        at = _run_app()
        _max_eval_samples_input(at).set_value(150).run(timeout=60)
        assert not at.exception
        assert "(n=200, evaluated on 150)" in _own_heading(at)

    def test_default_behaviour_analysis_is_actually_capped(self, synthetic_checkpoint):
        at = _run_app()
        _max_eval_samples_input(at).set_value(150).run(timeout=60)
        assert not at.exception
        counts_expander = next(exp for exp in at.expander if exp.label.startswith("Sample counts"))
        assert counts_expander.label == "Sample counts (total n=150)"


class TestSplitBy:
    """Each sub-strategy's own 1-2 "Split By" features define its
    prominent, implementable table+graph and "Decision variance explained"
    figure (see _render_substrategy)."""

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

    def test_variance_explained_metric_appears_once_split_by_is_set(self, synthetic_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")
        assert not any(m.label == "Decision variance explained by Split By features on claimed samples" for m in at.metric)

        at.multiselect(key=f"{c0}split_by").set_value(["street_norm"]).run(timeout=60)

        assert not at.exception
        metrics = [m for m in at.metric if m.label == "Decision variance explained by Split By features on claimed samples"]
        assert len(metrics) == 1
        assert metrics[0].value.endswith("%")

    def test_no_split_by_shows_a_hint_instead_of_a_metric(self, synthetic_checkpoint):
        at = _run_app()
        _add_substrategy(at, "root::")
        assert not at.exception
        assert any("Pick 1-2 Split By features" in c.value for c in at.caption)
        assert not any(m.label == "Decision variance explained by Split By features on claimed samples" for m in at.metric)


def _feature_table(at: AppTest, key_prefix: str = "root::"):
    """The per-sub-strategy feature-importance-and-interaction table (see
    _render_substrategy) -- opt-in behind its own "Show feature importance
    and interaction table" checkbox (computationally expensive), so this
    checks it first if it isn't already, then picks the table out of every
    st.dataframe on the page (there are others: _render_table's own
    summary/counts tables) by its distinctive column set."""
    checkbox = at.checkbox(key=f"{key_prefix}show_importance_table")
    if not checkbox.value:
        checkbox.set_value(True).run(timeout=60)
    return next(
        df.value for df in at.dataframe
        if list(df.value.columns) == ["Feature", "Importance", "Interaction with current Split By"]
    )


class TestFeatureTable:
    """The feature-importance-and-interaction table every sub-strategy
    page shows above "Suggested sub-strategies" (see _render_substrategy) --
    one row per display feature, always computed over this node's own
    default_df (see _decision_variance_by_key), so it's always scoped to
    exactly the sample currently matching this sub-strategy's own filters."""

    def test_present_with_a_row_per_feature(self, synthetic_checkpoint):
        at = _run_app()
        table = _feature_table(at)
        assert set(table["Feature"]) == {cfr_features.feature_label(k) for k in _FEATURE_KEYS}

    def test_interaction_column_is_a_dash_with_no_split_by_chosen(self, synthetic_checkpoint):
        at = _run_app()
        table = _feature_table(at)
        assert set(table["Interaction with current Split By"]) == {"—"}

    def test_interaction_column_is_a_dash_for_the_chosen_split_by_feature_itself(self, synthetic_checkpoint):
        at = _run_app()
        at.multiselect(key="root::split_by").set_value(["street_norm"]).run(timeout=60)
        table = _feature_table(at)
        own_row = table[table["Feature"] == cfr_features.feature_label("street_norm")].iloc[0]
        assert own_row["Interaction with current Split By"] == "—"

    def test_interaction_column_is_populated_for_other_features_once_split_by_is_chosen(self, synthetic_checkpoint):
        at = _run_app()
        at.multiselect(key="root::split_by").set_value(["street_norm"]).run(timeout=60)
        table = _feature_table(at)
        other_row = table[table["Feature"] == cfr_features.feature_label("hole_suited")].iloc[0]
        assert other_row["Interaction with current Split By"] != "—"
        assert other_row["Interaction with current Split By"].endswith("%")

    def test_interaction_column_is_populated_for_other_features_when_split_by_is_exact_hole_hand_alone(
        self, grid_joint_interaction_checkpoint,
    ):
        # Exact Hole Hand fills both of Split By's own slots by itself
        # (see _resolve_split_by) -- but that's a UI/memorability
        # constraint on what a person could pick, not a limit on what
        # _group_labels_for_rows can jointly group: (grid cell,
        # street_norm's own value) is just as well-defined a joint
        # grouping as any other pair. street_norm carries strong, real
        # signal here (see _grid_joint_interaction_checkpoint_path), so a
        # merely-not-"—" check wouldn't tell a genuine joint computation
        # apart from one wrongly forced to a flat 0% -- asserting a
        # clearly nonzero reading does.
        at = _run_app()
        at.multiselect(key="root::split_by").set_value(["hole_hand_grid_x_norm"]).run(timeout=60)
        assert not at.exception
        table = _feature_table(at)
        street_row = table[table["Feature"] == cfr_features.feature_label("street_norm")].iloc[0]
        interaction = street_row["Interaction with current Split By"]
        assert interaction != "—"
        assert int(interaction.rstrip("%")) >= 5

    def test_interaction_column_is_a_dash_for_exact_hole_hands_own_row_when_it_is_the_split_by(
        self, all_preflop_hole_hand_grid_checkpoint,
    ):
        at = _run_app()
        at.multiselect(key="root::split_by").set_value(["hole_hand_grid_x_norm"]).run(timeout=60)
        assert not at.exception
        table = _feature_table(at)
        own_row = table[table["Feature"] == "Exact Hole Hand"].iloc[0]
        assert own_row["Interaction with current Split By"] == "—"

    def test_interaction_column_differs_per_candidate_with_two_split_by_features(self, three_way_checkpoint):
        # Regression coverage for a real bug: with 2 Split By features
        # already chosen, _group_labels_for_rows' own general branch used
        # to combine only the *first two* of however many group_by_keys it
        # was given, silently dropping any 3rd -- so every remaining
        # candidate's own "interaction with current Split By" collapsed to
        # re-measuring the *same* 2-key grouping regardless of which
        # candidate was actually asked about, reading identically for
        # every row. connected_flop carries real signal here and
        # is_aggressor_previous_street doesn't (see _make_three_way_checkpoint) -- their
        # own values should differ.
        at = _run_app()
        at.multiselect(key="root::split_by").set_value(["street_norm", "hole_suited"]).run(timeout=60)
        assert not at.exception

        table = _feature_table(at)
        connected_value = table[table["Feature"] == cfr_features.feature_label("connected_flop")].iloc[0][
            "Interaction with current Split By"
        ]
        aggressor_value = table[table["Feature"] == cfr_features.feature_label("is_aggressor_previous_street")].iloc[0][
            "Interaction with current Split By"
        ]
        assert connected_value != "—"
        assert aggressor_value != "—"
        assert connected_value != aggressor_value

    def test_includes_exact_hole_hand_when_the_substrategy_is_100_percent_preflop(
        self, all_preflop_hole_hand_grid_checkpoint,
    ):
        at = _run_app()
        table = _feature_table(at)
        assert "Exact Hole Hand" in set(table["Feature"])

    def test_excludes_exact_hole_hand_when_rows_are_mixed_streets(self, hole_hand_grid_checkpoint):
        at = _run_app()
        table = _feature_table(at)
        assert "Exact Hole Hand" not in set(table["Feature"])

    def test_exact_hole_hand_is_sorted_into_its_own_ranked_position_not_appended_last(
        self, grid_outranks_other_checkpoint,
    ):
        # Regression coverage: the table used to filter Exact Hole Hand out
        # and unconditionally re-append it as the very last row, regardless
        # of its actual importance (see _render_substrategy's table_keys).
        # Here it has real, strongly-weighted signal and is_aggressor_previous_street has
        # none (see _grid_outranks_other_checkpoint_path), so it should
        # rank clearly above is_aggressor_previous_street, not below it.
        at = _run_app()
        table = _feature_table(at)
        assert list(table["Feature"]) == ["Exact Hole Hand", "Last Aggressor - Previous Street"]


class TestNavigation:
    """The sidebar's nested outline of the current sub-strategy tree
    doubles as the switcher that controls which single node's own content
    shows in the central column (see _render_navigation/_render_substrategy)
    -- clicking a node there re-renders the page to show it, rather than
    scrolling to it."""

    def test_root_button_present_and_selected_by_default(self, synthetic_checkpoint):
        at = _run_app()
        assert not at.exception
        root_btn = at.sidebar.button(key="nav::root::")
        assert root_btn.label == "Overall Strategy"
        assert root_btn.proto.type == "primary"

    def test_child_gets_its_own_indented_nav_button_and_becomes_selected(self, synthetic_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")  # auto-selected

        child_btn = at.sidebar.button(key=f"nav::{c0}")
        assert _nav_indent_px(at, c0) > _nav_indent_px(at, "root::")
        assert child_btn.proto.type == "primary"
        assert at.sidebar.button(key="nav::root::").proto.type == "secondary"

    def test_siblings_share_the_same_indent_level(self, synthetic_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")
        _select(at, "root::")
        c1 = _add_substrategy(at, "root::")

        assert _nav_indent_px(at, c0) == _nav_indent_px(at, c1)
        assert _nav_indent_px(at, c0) > _nav_indent_px(at, "root::")

    def test_grandchild_is_indented_one_level_deeper_than_its_parent(self, synthetic_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")
        _select(at, "root::")
        _add_substrategy(at, "root::")
        _select(at, c0)
        grandchild = _add_substrategy(at, c0)

        assert _nav_indent_px(at, grandchild) > _nav_indent_px(at, c0)

    def test_unclaimed_child_label_says_unclaimed(self, synthetic_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")
        assert at.sidebar.button(key=f"nav::{c0}").label == "Sub-strategy (unclaimed)"

    def test_child_claim_shows_up_in_its_nav_label(self, synthetic_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")
        at.multiselect(key=f"{c0}claim_filter_keys").set_value(["street_norm"]).run(timeout=60)
        observed = at.multiselect(key=f"{c0}claim_filter_values::street_norm").value
        _batch_set(at, {f"{c0}claim_filter_values::street_norm": observed[:1]})

        assert observed[0] in at.sidebar.button(key=f"nav::{c0}").label

    def test_long_claim_description_gets_truncated_in_nav_but_not_in_heading(self, synthetic_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")
        at.multiselect(key=f"{c0}claim_filter_keys").set_value(["hand_category_norm"]).run(timeout=60)

        assert not at.exception
        observed = at.multiselect(key=f"{c0}claim_filter_values::hand_category_norm").value
        full_desc = f"Hand Strength Tier = {', '.join(observed)}"
        assert full_desc in _own_heading(at)  # the real heading is never truncated

        nav_label = at.sidebar.button(key=f"nav::{c0}").label
        if len(full_desc) > 40:
            assert nav_label.rstrip().endswith("…")
            assert full_desc not in nav_label
        else:
            assert full_desc in nav_label

    def test_clicking_a_nav_button_switches_the_central_view(self, synthetic_checkpoint):
        at = _run_app()
        _add_substrategy(at, "root::")  # navigates away from root

        at.sidebar.button(key="nav::root::").click().run(timeout=60)

        assert not at.exception
        assert at.session_state["selected_substrategy"] == "root::"
        assert _own_heading(at).startswith("Overall Strategy")

    def test_only_the_selected_nodes_own_controls_are_present(self, synthetic_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")  # auto-selected
        assert not any(ms.key == "root::claim_filter_keys" for ms in at.multiselect)
        assert any(ms.key == f"{c0}claim_filter_keys" for ms in at.multiselect)

        at.sidebar.button(key="nav::root::").click().run(timeout=60)

        assert not at.exception
        assert any(ms.key == "root::claim_filter_keys" for ms in at.multiselect)
        assert not any(ms.key == f"{c0}claim_filter_keys" for ms in at.multiselect)

    def test_back_to_parent_button_switches_selection_upward(self, synthetic_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")

        at.button(key=f"{c0}back").click().run(timeout=60)

        assert not at.exception
        assert at.session_state["selected_substrategy"] == "root::"
        assert _own_heading(at).startswith("Overall Strategy")

    def test_central_quick_jump_button_switches_to_a_child(self, synthetic_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")
        child_id = c0.rsplit("substrategy_", 1)[1].rstrip(":")
        _select(at, "root::")

        jump_btn = at.button(key=f"root::jump::{child_id}")
        jump_btn.click().run(timeout=60)

        assert not at.exception
        assert at.session_state["selected_substrategy"] == c0

    def test_removing_the_selected_node_redirects_to_its_parent(self, synthetic_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")

        at.button(key=f"{c0}remove").click().run(timeout=60)

        assert not at.exception
        assert at.session_state["selected_substrategy"] == "root::"
        assert _own_heading(at).startswith("Overall Strategy")

    def test_removing_an_ancestor_of_the_selected_node_also_redirects(self, synthetic_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")
        grandchild = _add_substrategy(at, c0)
        assert at.session_state["selected_substrategy"] == grandchild

        _select(at, c0)
        at.button(key=f"{c0}remove").click().run(timeout=60)

        assert not at.exception
        assert at.session_state["selected_substrategy"] == "root::"


class TestStickyStateAcrossNavigation:
    """Regression coverage for a real bug found during development: since
    only the currently selected node's own widgets render each script run
    (see _render_substrategy), Streamlit forgets a *widget's own*
    session_state the instant its st.xxx(key=...) call stops executing for
    even one run -- confirmed directly against a fresh AppTest session,
    not just this app's own code. Left unhandled, navigating away from a
    node and back would silently reset its claim filters (breaking the
    waterfall row-claiming math for every sibling/descendant that depends
    on it) and its Split By/extras choices (a confusing UI regression).
    cfr_explorer.py fixes this with a set of small sticky-storage helpers
    (_local_filters_from_state/_split_by_from_state/_sticky_multiselect)
    that persist a widget's current value in a plain, non-widget
    session_state dict the widget's own render always keeps up to date."""

    def test_claim_filter_survives_navigating_away_and_back(self, synthetic_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")
        at.multiselect(key=f"{c0}claim_filter_keys").set_value(["street_norm"]).run(timeout=60)
        observed = at.multiselect(key=f"{c0}claim_filter_values::street_norm").value
        _batch_set(at, {f"{c0}claim_filter_values::street_norm": observed[:1]})
        heading_before = _own_heading(at)

        _select(at, "root::")
        _select(at, c0)

        assert not at.exception
        assert _own_heading(at) == heading_before
        assert at.multiselect(key=f"{c0}claim_filter_keys").value == ["street_norm"]
        assert at.multiselect(key=f"{c0}claim_filter_values::street_norm").value == observed[:1]

    def test_first_siblings_claim_still_subtracted_after_revisiting_it(self, synthetic_checkpoint):
        # The exact scenario that caught the underlying bug: c0's own
        # claim must still be subtracted from what c1 sees, even after a
        # run where c0's own widgets didn't render at all.
        at = _run_app()
        c0 = _add_substrategy(at, "root::")
        at.multiselect(key=f"{c0}claim_filter_keys").set_value(["street_norm"]).run(timeout=60)
        _batch_set(at, {f"{c0}claim_filter_values::street_norm": ["Preflop"]})

        _select(at, "root::")
        c1 = _add_substrategy(at, "root::")
        at.multiselect(key=f"{c1}claim_filter_keys").set_value(["street_norm"]).run(timeout=60)
        _batch_set(at, {f"{c1}claim_filter_values::street_norm": ["Flop"]})
        # Revisit c0 (a no-op read, but exercises the exact code path that
        # broke -- rendering some *other* node after c0's own widgets have
        # gone a run without executing).
        _select(at, c0)
        _select(at, c1)

        assert not at.exception
        c1_n = int(re.search(r"n=([\d,]+)", _own_heading(at)).group(1).replace(",", ""))
        assert 0 < c1_n < 200  # nonzero, and strictly less than the parent's full pool

    def test_split_by_survives_navigating_away_and_back(self, synthetic_checkpoint):
        at = _run_app()
        c0 = _add_substrategy(at, "root::")
        at.multiselect(key=f"{c0}split_by").set_value(["street_norm"]).run(timeout=60)

        _select(at, "root::")
        _select(at, c0)

        assert not at.exception
        assert at.multiselect(key=f"{c0}split_by").value == ["street_norm"]

    def test_extra_graph_survives_navigating_away_and_back(self, synthetic_checkpoint):
        at = _run_app()
        at.multiselect(key="root::extra_graph").set_value(["street_norm"]).run(timeout=60)

        _add_substrategy(at, "root::")
        _select(at, "root::")

        assert not at.exception
        assert at.multiselect(key="root::extra_graph").value == ["street_norm"]
        assert len(at.get("plotly_chart")) == 1


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


_GRID_OUTRANKS_OTHER_FEATURE_KEYS = ("hole_hand_grid_x_norm", "hole_hand_grid_y_norm", "is_aggressor_previous_street")


@pytest.fixture(scope="module")
def _grid_outranks_other_checkpoint_path(tmp_path_factory) -> str:
    """All rows preflop, with Exact Hole Hand's own grid coordinates given
    a big net input weight (real signal, like hand_idx in
    _make_importance_per_level_checkpoint) and is_aggressor_previous_street's zeroed out
    entirely (guaranteed no effect, like hole_idx there) -- so Exact Hole
    Hand should rank clearly *above* is_aggressor_previous_street in the feature table.
    Regression coverage for the table's own sort: it used to filter Exact
    Hole Hand out and unconditionally re-append it last, regardless of its
    actual importance (see _render_substrategy's table_keys)."""
    path = os.path.join(str(tmp_path_factory.mktemp("cfr_explorer_grid_outranks_other")), "checkpoint")
    rng = np.random.default_rng(0)
    feature_dim = len(cfr_features.feature_indices(_GRID_OUTRANKS_OTHER_FEATURE_KEYS))
    x_idx = _GRID_OUTRANKS_OTHER_FEATURE_KEYS.index("hole_hand_grid_x_norm")
    y_idx = _GRID_OUTRANKS_OTHER_FEATURE_KEYS.index("hole_hand_grid_y_norm")
    aggressor_idx = _GRID_OUTRANKS_OTHER_FEATURE_KEYS.index("is_aggressor_previous_street")

    torch.manual_seed(0)  # AdvantageNet's init otherwise draws from torch's unseeded global RNG
    net = cfr_networks.AdvantageNet(input_dim=feature_dim, hidden_sizes=(8, 8))
    with torch.no_grad():
        net.hidden[0].block[0].weight[:, x_idx] *= 20.0
        net.hidden[0].block[0].weight[:, y_idx] *= 20.0
        net.hidden[0].block[0].weight[:, aggressor_idx] = 0.0
    net_config = cfr_networks.AdvantageNetConfig(
        feature_keys=_GRID_OUTRANKS_OTHER_FEATURE_KEYS, hidden_sizes=(8, 8), table_size=3,
    )
    cfr_networks.save(net, net_config, path)

    num_samples = 600
    reservoir = cfr_reservoir.ReservoirBuffer(
        capacity=num_samples, feature_dim=feature_dim, num_actions=strategy.NUM_ACTION_CATEGORIES, rng=rng,
    )
    for _ in range(num_samples):
        feats = rng.random(feature_dim).astype(np.float32)
        feats[x_idx] = rng.integers(0, 3) / 12.0
        feats[y_idx] = rng.integers(0, 3) / 12.0
        regrets = rng.normal(size=strategy.NUM_ACTION_CATEGORIES).astype(np.float32)
        legal_mask = rng.random(strategy.NUM_ACTION_CATEGORIES) > 0.3
        legal_mask[strategy.ACTION_CALL] = True
        reservoir.add(feats, regrets, legal_mask, float(rng.integers(1, 10)))
    reservoir.save(path)
    return path


@pytest.fixture
def grid_outranks_other_checkpoint(_grid_outranks_other_checkpoint_path, monkeypatch):
    monkeypatch.setenv("CFR_EXPLORER_CHECKPOINT_PATH", _grid_outranks_other_checkpoint_path)
    return _grid_outranks_other_checkpoint_path


_GRID_JOINT_INTERACTION_FEATURE_KEYS = ("hole_hand_grid_x_norm", "hole_hand_grid_y_norm", "street_norm")


@pytest.fixture(scope="module")
def _grid_joint_interaction_checkpoint_path(tmp_path_factory) -> str:
    """All rows preflop, street_norm given a big net input weight (real
    signal, like hand_idx in _make_importance_per_level_checkpoint) while
    Exact Hole Hand's own grid coordinates get *zero* weight (guaranteed
    no effect, like hole_idx there) and a small, unboosted range (3x3 = 9
    cells, not the full 169) so a joint (grid cell, street_norm level)
    grouping still has plenty of samples per group despite carrying no
    real signal of its own. Regression coverage for the feature table's
    "Interaction with current Split By" column actually computing the
    *joint* grouping instead of being forced to 0 whenever Exact Hole Hand
    is the current Split By (see _decision_variance_by_key): with
    street_norm's own strong, isolated signal, that joint computation
    should read clearly above 0% (empirically ~9%, some diluted by the
    grid's own finer partition -- see _decision_variance_explained's own
    leave-one-out correction), not just "not literally a dash"."""
    path = os.path.join(str(tmp_path_factory.mktemp("cfr_explorer_grid_joint_interaction")), "checkpoint")
    rng = np.random.default_rng(0)
    feature_dim = len(cfr_features.feature_indices(_GRID_JOINT_INTERACTION_FEATURE_KEYS))
    x_idx = _GRID_JOINT_INTERACTION_FEATURE_KEYS.index("hole_hand_grid_x_norm")
    y_idx = _GRID_JOINT_INTERACTION_FEATURE_KEYS.index("hole_hand_grid_y_norm")
    street_idx = _GRID_JOINT_INTERACTION_FEATURE_KEYS.index("street_norm")

    torch.manual_seed(0)  # AdvantageNet's init otherwise draws from torch's unseeded global RNG
    net = cfr_networks.AdvantageNet(input_dim=feature_dim, hidden_sizes=(8, 8))
    with torch.no_grad():
        net.hidden[0].block[0].weight[:, street_idx] *= 20.0
        net.hidden[0].block[0].weight[:, x_idx] = 0.0
        net.hidden[0].block[0].weight[:, y_idx] = 0.0
    net_config = cfr_networks.AdvantageNetConfig(
        feature_keys=_GRID_JOINT_INTERACTION_FEATURE_KEYS, hidden_sizes=(8, 8), table_size=3,
    )
    cfr_networks.save(net, net_config, path)

    num_samples = 600
    reservoir = cfr_reservoir.ReservoirBuffer(
        capacity=num_samples, feature_dim=feature_dim, num_actions=strategy.NUM_ACTION_CATEGORIES, rng=rng,
    )
    for _ in range(num_samples):
        feats = rng.random(feature_dim).astype(np.float32)
        feats[x_idx] = rng.integers(0, 3) / 12.0
        feats[y_idx] = rng.integers(0, 3) / 12.0
        if cfr_features.bucket_label("street_norm", feats[street_idx]) == "Preflop":
            feats[street_idx] = 0.0
        else:
            feats[street_idx] = 1.0
        regrets = rng.normal(size=strategy.NUM_ACTION_CATEGORIES).astype(np.float32)
        legal_mask = rng.random(strategy.NUM_ACTION_CATEGORIES) > 0.3
        legal_mask[strategy.ACTION_CALL] = True
        reservoir.add(feats, regrets, legal_mask, float(rng.integers(1, 10)))
    reservoir.save(path)
    return path


@pytest.fixture
def grid_joint_interaction_checkpoint(_grid_joint_interaction_checkpoint_path, monkeypatch):
    monkeypatch.setenv("CFR_EXPLORER_CHECKPOINT_PATH", _grid_joint_interaction_checkpoint_path)
    return _grid_joint_interaction_checkpoint_path


_MAX_SPLIT_EXCLUDES_GRID_FEATURE_KEYS = ("hole_suited", "is_aggressor_previous_street", "hole_hand_grid_x_norm", "hole_hand_grid_y_norm")


@pytest.fixture(scope="module")
def _max_split_excludes_hole_hand_grid_checkpoint_path(tmp_path_factory) -> str:
    """All rows preflop (Exact Hole Hand's own 100%-preflop Split By
    requirement holds -- see _hole_hand_grid_split_by_available), with two
    ordinary candidate features (hole_suited, is_aggressor_previous_street) alongside it --
    used to prove "Add maximum interaction split"/"Add maximum importance
    split" never treat Exact Hole Hand as an eligible candidate even when
    it's genuinely available (unlike "Add best second Split By feature"/
    "Optimise split by features", which the user explicitly asked to
    support it -- see _splittable_candidates' own include_hole_hand_grid)."""
    path = os.path.join(str(tmp_path_factory.mktemp("cfr_explorer_max_split_excludes_grid")), "checkpoint")
    rng = np.random.default_rng(0)
    feature_dim = len(cfr_features.feature_indices(_MAX_SPLIT_EXCLUDES_GRID_FEATURE_KEYS))
    net = cfr_networks.AdvantageNet(input_dim=feature_dim, hidden_sizes=(8, 8))
    net_config = cfr_networks.AdvantageNetConfig(
        feature_keys=_MAX_SPLIT_EXCLUDES_GRID_FEATURE_KEYS, hidden_sizes=(8, 8), table_size=3,
    )
    cfr_networks.save(net, net_config, path)
    num_samples = 200
    reservoir = cfr_reservoir.ReservoirBuffer(
        capacity=num_samples, feature_dim=feature_dim, num_actions=strategy.NUM_ACTION_CATEGORIES, rng=rng,
    )
    x_idx = _MAX_SPLIT_EXCLUDES_GRID_FEATURE_KEYS.index("hole_hand_grid_x_norm")
    y_idx = _MAX_SPLIT_EXCLUDES_GRID_FEATURE_KEYS.index("hole_hand_grid_y_norm")
    for _ in range(num_samples):
        feats = rng.random(feature_dim).astype(np.float32)
        feats[x_idx] = rng.integers(0, 13) / 12.0
        feats[y_idx] = rng.integers(0, 13) / 12.0
        regrets = rng.normal(size=strategy.NUM_ACTION_CATEGORIES).astype(np.float32)
        legal_mask = rng.random(strategy.NUM_ACTION_CATEGORIES) > 0.3
        legal_mask[strategy.ACTION_CALL] = True
        reservoir.add(feats, regrets, legal_mask, float(rng.integers(1, 10)))
    reservoir.save(path)
    return path


@pytest.fixture
def max_split_excludes_hole_hand_grid_checkpoint(_max_split_excludes_hole_hand_grid_checkpoint_path, monkeypatch):
    monkeypatch.setenv("CFR_EXPLORER_CHECKPOINT_PATH", _max_split_excludes_hole_hand_grid_checkpoint_path)
    return _max_split_excludes_hole_hand_grid_checkpoint_path


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

    def test_add_best_second_split_by_never_picks_it_even_when_it_is_the_only_candidate(
        self, all_preflop_hole_hand_grid_checkpoint,
    ):
        # Exact Hole Hand always fills *both* Split By slots by itself (see
        # _resolve_split_by) -- with street_norm (this checkpoint's only
        # ordinary feature, pinned constant -- see
        # _all_preflop_hole_hand_grid_checkpoint_path) already occupying
        # the one slot this button is trying to fill, there's no room left
        # for it: picking it wouldn't add a second dimension to the
        # existing rule, it would silently discard street_norm instead.
        # Unlike "Optimise split by features" (which searches fresh, so
        # picking it alone is legitimate there), this button must no-op
        # rather than fall back to it.
        at = _run_app()
        at.multiselect(key="root::split_by").set_value(["street_norm"]).run(timeout=60)
        at.button(key="root::add_best_second_split_by").click().run(timeout=60)

        assert not at.exception
        assert at.multiselect(key="root::split_by").value == ["street_norm"]

    def test_optimise_split_by_can_pick_it_alone_as_the_winning_pair(
        self, all_preflop_hole_hand_grid_checkpoint,
    ):
        # With street_norm constant (see above), there are zero genuine
        # 2-feature pairs available -- "Optimise split by features" must
        # still not no-op here, since Exact Hole Hand alone is itself a
        # valid (better than nothing) pick (see _add_optimise_split_by's
        # own solo-evaluation branch).
        at = _run_app()
        at.button(key="root::optimise_split_by").click().run(timeout=60)

        assert not at.exception
        assert at.multiselect(key="root::split_by").value == ["hole_hand_grid_x_norm"]

    def test_max_interaction_split_never_picks_it(self, max_split_excludes_hole_hand_grid_checkpoint):
        # Regression guard for the user's explicit scope choice: unlike
        # "Add best second Split By feature"/"Optimise split by features",
        # this button must never treat Exact Hole Hand as an eligible
        # candidate (it would create up to 169 children, one per grid
        # cell -- see _splittable_candidates' own include_hole_hand_grid,
        # which _add_max_interaction_split doesn't pass). is_aggressor_previous_street is
        # the sole legitimate ordinary candidate left once hole_suited is
        # the current Split By pick, so picking it deterministically
        # (rather than Exact Hole Hand, which is also genuinely available
        # here -- every row is preflop) proves the exclusion held.
        at = _run_app()
        at.multiselect(key="root::split_by").set_value(["hole_suited"]).run(timeout=60)
        at.button(key="root::add_max_interaction_split").click().run(timeout=60)

        assert not at.exception
        children = at.session_state["substrategy_children"]["root::"]
        claims = at.session_state["substrategy_claims"]
        claimed_features = {next(iter(claims[f"root::substrategy_{cid}::"])) for cid in children}
        assert claimed_features == {"is_aggressor_previous_street"}

    def test_max_importance_split_never_picks_it(self, max_split_excludes_hole_hand_grid_checkpoint):
        # Same guard, for "Add maximum importance split" -- here with no
        # Split By chosen yet, so both hole_suited and is_aggressor_previous_street are
        # legitimate ordinary candidates alongside Exact Hole Hand (also
        # genuinely available -- every row is preflop); the winner must
        # still be one of the two ordinary features, never the grid key.
        at = _run_app()
        at.button(key="root::add_max_importance_split").click().run(timeout=60)

        assert not at.exception
        children = at.session_state["substrategy_children"]["root::"]
        claims = at.session_state["substrategy_claims"]
        claimed_features = {next(iter(claims[f"root::substrategy_{cid}::"])) for cid in children}
        assert claimed_features <= {"hole_suited", "is_aggressor_previous_street"}


_GROUP_RELATIVE_FEATURE_KEYS = ("hole_hand_grid_x_norm", "hole_hand_grid_y_norm", "hole_suited")


def _make_group_relative_checkpoint(path: str, rng: np.random.Generator) -> None:
    """A checkpoint whose feature set includes Exact Hole Hand alongside
    `hole_suited`, deterministically derived from Exact Hole Hand's own
    grid position (upper-right of the diagonal, matching the real "suited"
    convention -- see features.hole_hand_grid_label) -- every row preflop,
    so Exact Hole Hand's Split By is available with no claim filter first
    needed. Regression coverage for "a sub-strategy's own importance view
    should assume its parent's own Split By grouping is already priced in"
    (see cfr_explorer._group_labels_for_rows/_decision_variance_explained/
    _render_substrategy): once a parent sub-strategy splits by Exact Hole
    Hand, hole_suited -- fully determined by it -- should explain exactly
    0% decision variance for any sub-strategy underneath, instead of
    sharing credit for the same signal."""
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


class TestGroupRelativeImportance:
    """A sub-strategy's own importance view assumes its *parent's* own
    Split By grouping is already priced in (see _group_labels_for_rows/
    _decision_variance_explained). In this checkpoint's synthetic data,
    Exact Hole Hand fully determines hole_suited, so once a parent splits
    by Exact Hole Hand, a child underneath it should show exactly 0%
    decision variance explained for hole_suited instead of sharing credit
    for the same signal Exact Hole Hand already explains."""

    def test_hole_suited_shows_nonzero_importance_with_no_parent_grouping(self, group_relative_checkpoint):
        at = _run_app()
        assert not at.exception
        option = next(o for o in at.multiselect(key="root::split_by").options if o.startswith("Suited Hole Cards"))
        value = float(re.search(r"\((\d+)% variance explained\)", option).group(1))
        assert value > 0.0

    def test_hole_suited_drops_to_exactly_zero_under_an_exact_hole_hand_parent_split(
        self, group_relative_checkpoint,
    ):
        at = _run_app()
        at.multiselect(key="root::split_by").set_value(["hole_hand_grid_x_norm"]).run(timeout=60)
        c0 = _add_substrategy(at, "root::")

        assert not at.exception
        option = next(o for o in at.multiselect(key=f"{c0}split_by").options if o.startswith("Suited Hole Cards"))
        assert option.endswith("(0% variance explained)")

    def test_a_grandchild_under_an_unrelated_parent_split_keeps_a_nonzero_importance(
        self, group_relative_checkpoint,
    ):
        # Sanity check for the other direction: a child added under a
        # parent that has *not* split by Exact Hole Hand (root's own
        # default, unset Split By) should see hole_suited's ordinary,
        # nonzero importance -- confirming the zeroing above is
        # specifically a consequence of the parent's own grouping, not
        # some unconditional side effect of merely being a non-root node.
        at = _run_app()
        c0 = _add_substrategy(at, "root::")

        assert not at.exception
        option = next(o for o in at.multiselect(key=f"{c0}split_by").options if o.startswith("Suited Hole Cards"))
        value = float(re.search(r"\((\d+)% variance explained\)", option).group(1))
        assert value > 0.0


_INTERACTION_FEATURE_KEYS = ("hole_suited", "street_norm")


def _make_interaction_checkpoint(path: str, interaction: bool, n: int = 8000, seed: int = 0) -> None:
    """A checkpoint whose net is trained on a signal (isolated to a single
    action category, so regret-matching doesn't blur it across several
    near-tied outputs) that's a function of `hole_suited` -- REVERSED
    specifically on the Flop street when `interaction` is True, identical
    across every street when it's False.

    Regression coverage for "Decision variance explained by Split By
    features on claimed samples" (see cfr_explorer._decision_variance_explained):
    a child sub-strategy claiming Flop, sharing its parent's own Split By
    (hole_suited), should read meaningfully above 0% when `interaction` is
    True -- the parent's own (population-wide, blended-across-streets)
    per-hole_suited baseline does NOT correctly predict Flop's own
    (reversed) relationship, so grouping the Flop-only residual by
    hole_suited again explains real, additional variance -- and should
    read near 0% when `interaction` is False, since there the parent's own
    baseline already predicts Flop's behavior just as well as any other
    street's."""
    feature_dim = len(cfr_features.feature_indices(_INTERACTION_FEATURE_KEYS))
    suited_idx = _INTERACTION_FEATURE_KEYS.index("hole_suited")
    street_idx = _INTERACTION_FEATURE_KEYS.index("street_norm")
    rng = np.random.default_rng(seed)
    suited = (rng.random(n) < 0.5).astype(np.float32)
    street_choice = rng.integers(0, 4, size=n)
    street = np.array([0.0, 1 / 3, 2 / 3, 1.0], dtype=np.float32)[street_choice]
    is_flop = street_choice == 1
    signal = np.where(is_flop, -20.0 * suited, 20.0 * suited) if interaction else 20.0 * suited

    X = np.zeros((n, feature_dim), dtype=np.float32)
    X[:, suited_idx] = suited
    X[:, street_idx] = street
    y = np.zeros((n, strategy.NUM_ACTION_CATEGORIES), dtype=np.float32)
    y[:, 0] = signal + rng.normal(scale=0.1, size=n)  # isolate the signal to one action
    y[:, 1:] = rng.normal(scale=0.1, size=(n, strategy.NUM_ACTION_CATEGORIES - 1))

    torch.manual_seed(0)
    net = cfr_networks.AdvantageNet(input_dim=feature_dim, hidden_sizes=(32, 32), dropout=0.0)
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    net.train()
    train_rng = np.random.default_rng(0)
    for _ in range(2000):
        idx = train_rng.integers(0, n, size=256)
        pred = net(Xt[idx])
        loss = ((pred - yt[idx]) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    net.eval()

    net_config = cfr_networks.AdvantageNetConfig(
        feature_keys=_INTERACTION_FEATURE_KEYS, hidden_sizes=(32, 32), table_size=3,
    )
    cfr_networks.save(net, net_config, path)

    reservoir = cfr_reservoir.ReservoirBuffer(
        capacity=n, feature_dim=feature_dim, num_actions=strategy.NUM_ACTION_CATEGORIES, rng=rng,
    )
    for i in range(n):
        regrets = rng.normal(size=strategy.NUM_ACTION_CATEGORIES).astype(np.float32)
        legal_mask = rng.random(strategy.NUM_ACTION_CATEGORIES) > 0.3
        legal_mask[strategy.ACTION_CALL] = True
        reservoir.add(X[i], regrets, legal_mask, float(rng.integers(1, 10)))
    reservoir.save(path)


@pytest.fixture(scope="module")
def _interaction_checkpoint_path(tmp_path_factory) -> str:
    path = os.path.join(str(tmp_path_factory.mktemp("cfr_explorer_interaction")), "checkpoint")
    _make_interaction_checkpoint(path, interaction=True)
    return path


@pytest.fixture
def interaction_checkpoint(_interaction_checkpoint_path, monkeypatch):
    monkeypatch.setenv("CFR_EXPLORER_CHECKPOINT_PATH", _interaction_checkpoint_path)
    return _interaction_checkpoint_path


@pytest.fixture(scope="module")
def _no_interaction_checkpoint_path(tmp_path_factory) -> str:
    path = os.path.join(str(tmp_path_factory.mktemp("cfr_explorer_no_interaction")), "checkpoint")
    _make_interaction_checkpoint(path, interaction=False)
    return path


@pytest.fixture
def no_interaction_checkpoint(_no_interaction_checkpoint_path, monkeypatch):
    monkeypatch.setenv("CFR_EXPLORER_CHECKPOINT_PATH", _no_interaction_checkpoint_path)
    return _no_interaction_checkpoint_path


def _decision_variance_metric_value(at: AppTest) -> int:
    metric = next(m for m in at.metric if "Decision variance" in m.label)
    return int(metric.value.rstrip("%"))


class TestDecisionVarianceExplained:
    """Regression coverage for a real bug reported in production use: a
    child sub-strategy sharing its parent's own Split By feature, but with
    its own claim filter narrowing it to a subset, always read 0% "SHAP
    explained by Split By features" -- misleading, since a narrower rule
    over that same feature can genuinely capture real, additional signal
    the parent's broader-population analysis doesn't. cfr_explorer.
    _decision_variance_explained replaces that SHAP-ratio computation with
    a direct ANOVA-style "variance explained by grouping" statistic
    (renamed "Decision variance explained by Split By features on claimed
    samples" to match), computed against a baseline drawn from the
    *parent's* own full claimed scope -- see that function's own
    docstring."""

    def test_shared_split_by_reads_meaningfully_above_zero_with_genuine_interaction(
        self, interaction_checkpoint,
    ):
        at = _run_app()
        at.multiselect(key="root::split_by").set_value(["hole_suited"]).run(timeout=60)
        c0 = _add_substrategy(at, "root::")
        at.multiselect(key=f"{c0}claim_filter_keys").set_value(["street_norm"]).run(timeout=60)
        _batch_set(at, {f"{c0}claim_filter_values::street_norm": ["Flop"]})

        assert not at.exception
        assert at.multiselect(key=f"{c0}split_by").value == ["hole_suited"]  # inherited, same as the parent's
        assert _decision_variance_metric_value(at) >= 20

    def test_shared_split_by_reads_near_zero_with_no_interaction(self, no_interaction_checkpoint):
        at = _run_app()
        at.multiselect(key="root::split_by").set_value(["hole_suited"]).run(timeout=60)
        c0 = _add_substrategy(at, "root::")
        at.multiselect(key=f"{c0}claim_filter_keys").set_value(["street_norm"]).run(timeout=60)
        _batch_set(at, {f"{c0}claim_filter_values::street_norm": ["Flop"]})

        assert not at.exception
        assert at.multiselect(key=f"{c0}split_by").value == ["hole_suited"]
        assert _decision_variance_metric_value(at) <= 10

    def test_root_itself_has_no_parent_adjustment(self, interaction_checkpoint):
        # Root has no parent, so its own metric is plain, unadjusted
        # variance explained -- just exercised here for "doesn't error and
        # produces a sane percentage", since root's own claimed rows
        # aren't narrowed the way a child's are in the tests above.
        at = _run_app()
        at.multiselect(key="root::split_by").set_value(["hole_suited"]).run(timeout=60)

        assert not at.exception
        value = _decision_variance_metric_value(at)
        assert 0 <= value <= 100


class TestStrategySaveLoad:
    """Save/load for a sub-strategy tree's own structure -- which nodes
    exist and in what order, each one's own claim filters, and each one's
    own Split By pick (deliberately not the "purely illustrative" Add
    table/Add graph extras -- see _current_strategy_state) -- plus an
    always-on autosave/autoload of that exact same shape, so a session
    resumes right where it left off with no deliberate action required."""

    def _autosave_path(self, isolated_strategies_dir: str) -> str:
        return os.path.join(isolated_strategies_dir, "_autosave.json")

    def test_nothing_is_autosaved_before_any_change_is_made(self, synthetic_checkpoint, isolated_strategies_dir):
        _run_app()
        assert not os.path.exists(self._autosave_path(isolated_strategies_dir))

    def test_adding_a_substrategy_triggers_an_autosave(self, synthetic_checkpoint, isolated_strategies_dir):
        at = _run_app()
        _add_substrategy(at, "root::")

        autosave_path = self._autosave_path(isolated_strategies_dir)
        assert os.path.exists(autosave_path)
        with open(autosave_path) as f:
            state = json.load(f)
        assert state["children"]["root::"] == at.session_state["substrategy_children"]["root::"]

    def test_a_fresh_app_instance_resumes_the_previous_sessions_autosave(
        self, synthetic_checkpoint, isolated_strategies_dir,
    ):
        at = _run_app()
        child_prefix = _add_substrategy(at, "root::")
        at.multiselect(key=f"{child_prefix}claim_filter_keys").set_value(["street_norm"]).run(timeout=60)
        _batch_set(at, {f"{child_prefix}claim_filter_values::street_norm": ["Preflop"]})

        at2 = _run_app()  # a brand new session, sharing only the checkpoint + strategies dir on disk

        assert not at2.exception
        assert at2.session_state["substrategy_children"]["root::"] == at.session_state["substrategy_children"]["root::"]
        assert at2.session_state["substrategy_claims"][child_prefix] == {"street_norm": ["Preflop"]}

    def test_autoload_restores_the_previously_selected_node(self, synthetic_checkpoint, isolated_strategies_dir):
        at = _run_app()
        child_prefix = _add_substrategy(at, "root::")  # also selects the new child
        assert at.session_state["selected_substrategy"] == child_prefix

        at2 = _run_app()
        assert at2.session_state["selected_substrategy"] == child_prefix

    def test_save_and_load_round_trips_a_named_strategy(self, synthetic_checkpoint, isolated_strategies_dir):
        at = _run_app()
        child_prefix = _add_substrategy(at, "root::")
        at.multiselect(key=f"{child_prefix}claim_filter_keys").set_value(["street_norm"]).run(timeout=60)
        # The claim filter's own "keep values" multiselect and the "Save
        # as" text input are set together in one .run() (rather than the
        # text input getting its own later .run()) -- see _batch_set's own
        # docstring: a format_func-based multiselect (every "keep values"
        # widget included) isn't reliably preserved by AppTest across a
        # run where it's not re-asserted alongside whatever else changed.
        at.multiselect(key=f"{child_prefix}claim_filter_values::street_norm").set_value(["Preflop"])
        at.text_input(key="strategy_save_name").set_value("My Strategy")
        at.run(timeout=60)
        at.button(key="strategy_save_button").click().run(timeout=60)

        # Diverge from what was saved: remove the child sub-strategy (its
        # own "remove" button lives on its own page -- see
        # _render_substrategy -- and _add_substrategy already left it
        # selected).
        at.button(key=f"{child_prefix}remove").click().run(timeout=60)
        assert at.session_state["substrategy_children"]["root::"] == []

        at.selectbox(key="strategy_load_choice").set_value("My Strategy").run(timeout=60)
        at.button(key="strategy_load_button").click().run(timeout=60)

        assert not at.exception
        assert at.session_state["substrategy_children"]["root::"] == [
            child_prefix.rstrip(":").rsplit("substrategy_", 1)[-1]
        ]
        assert at.session_state["substrategy_claims"][child_prefix] == {"street_norm": ["Preflop"]}

    def test_saved_strategy_appears_as_a_load_option(self, synthetic_checkpoint, isolated_strategies_dir):
        at = _run_app()
        at.text_input(key="strategy_save_name").set_value("Tight range").run(timeout=60)
        at.button(key="strategy_save_button").click().run(timeout=60)

        assert "Tight range" in at.selectbox(key="strategy_load_choice").options

    def test_save_button_disabled_until_a_name_is_entered(self, synthetic_checkpoint, isolated_strategies_dir):
        at = _run_app()
        assert at.button(key="strategy_save_button").disabled

        at.text_input(key="strategy_save_name").set_value("   ").run(timeout=60)
        assert at.button(key="strategy_save_button").disabled  # blank after stripping -- still disabled

        at.text_input(key="strategy_save_name").set_value("Loose range").run(timeout=60)
        assert not at.button(key="strategy_save_button").disabled

    def test_load_button_disabled_until_a_saved_strategy_is_chosen(self, synthetic_checkpoint, isolated_strategies_dir):
        at = _run_app()
        assert at.button(key="strategy_load_button").disabled

    def test_save_name_is_sanitized_against_directory_traversal(
        self, synthetic_checkpoint, isolated_strategies_dir,
    ):
        at = _run_app()
        at.text_input(key="strategy_save_name").set_value("../../evil").run(timeout=60)
        at.button(key="strategy_save_button").click().run(timeout=60)

        assert not at.exception
        # The traversal characters are stripped, not honored -- the file
        # lands inside the isolated strategies dir itself, nowhere above it.
        assert os.listdir(isolated_strategies_dir) == ["evil.json"]

    def test_illustrative_extras_are_not_part_of_the_saved_state(self, synthetic_checkpoint, isolated_strategies_dir):
        # Add table/Add graph are scratch analysis, not part of a
        # sub-strategy's own implementable rule -- see the module docstring
        # and _current_strategy_state -- so they're deliberately excluded
        # from what gets saved.
        at = _run_app()
        at.multiselect(key="root::extra_table").set_value(["hole_suited"]).run(timeout=60)
        at.text_input(key="strategy_save_name").set_value("No extras").run(timeout=60)
        at.button(key="strategy_save_button").click().run(timeout=60)

        with open(os.path.join(isolated_strategies_dir, "No extras.json")) as f:
            state = json.load(f)
        assert set(state.keys()) == {"children", "claims", "split_by", "selected"}

