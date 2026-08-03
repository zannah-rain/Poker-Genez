"""Interactive Streamlit app for interrogating a trained Single Deep CFR
strategy: mark any of the checkpoint's features as a filter, a "group
split" (renders a separate table per observed combination of values), or a
"table split" (becomes a row/column axis within each table -- capped at 2,
since a table only has two axes), then see the current net's average action
distribution over the matching reservoir samples.

Run with:
    streamlit run poker_ga/cfr_explorer.py -- --checkpoint-path cfr_runs/checkpoint_latest

(everything after `--` is this script's own argv; the sidebar also lets you
change/reload a different checkpoint path at runtime, and how many
reservoir samples to load).

Action probabilities shown are the *current* net's regret-matching
strategy for each sampled situation -- not the regret values stored in the
reservoir itself, which reflect whatever earlier version of the net
collected them. The reservoir just supplies a realistic, CFR-visitation-
weighted sample of situations to ask the current net about.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import streamlit as st
import torch

import cfr_actions
import cfr_features
import cfr_networks
import cfr_reservoir
import strategy

DEFAULT_CHECKPOINT_PATH = os.path.join("cfr_runs", "checkpoint_latest")
DEFAULT_MAX_SAMPLES = 50_000

_FEATURE_COL_PREFIX = "feat::"
_ACTION_COL_PREFIX = "action::"

ROLE_UNUSED = "Unused"
ROLE_FILTER = "Filter"
ROLE_GROUP_SPLIT = "Group split"
ROLE_TABLE_SPLIT = "Table split"
ROLES = (ROLE_UNUSED, ROLE_FILTER, ROLE_GROUP_SPLIT, ROLE_TABLE_SPLIT)
MAX_TABLE_SPLIT_FEATURES = 2

_COLLAPSED_LABELS = ("Fold", "Call", "Raise", "All-In")
_COLLAPSED_GROUP_OF_ACTION = {
    strategy.ACTION_FOLD: "Fold",
    strategy.ACTION_CALL: "Call",
    **{a: "Raise" for a in strategy.RAISE_ACTIONS},
    strategy.ACTION_ALLIN: "All-In",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str, default=DEFAULT_CHECKPOINT_PATH)
    args, _unknown = parser.parse_known_args()
    return args


def _default_checkpoint_path() -> str:
    # The env var is purely so tests/test_cfr_explorer.py's AppTest-based
    # tests can point at a synthetic checkpoint -- AppTest.from_file has no
    # way to pass CLI args the way `streamlit run script -- args` does.
    return os.environ.get("CFR_EXPLORER_CHECKPOINT_PATH") or _parse_args().checkpoint_path


@st.cache_resource(show_spinner="Loading checkpoint...")
def _load_checkpoint(checkpoint_path: str):
    net, net_config = cfr_networks.load(checkpoint_path)
    reservoir = cfr_reservoir.ReservoirBuffer.load(checkpoint_path, rng=np.random.default_rng(0))
    return net, net_config, reservoir


@st.cache_resource(show_spinner="Computing the current strategy over reservoir samples...")
def _build_dataframe(checkpoint_path: str, max_samples: int) -> tuple[pd.DataFrame, np.ndarray]:
    """(df, raw_features): df has one row per (subsampled) reservoir entry
    -- one Categorical column per *displayed* feature (its bucket label, in
    the feature's own natural value order -- see
    cfr_features.bucket_categories; a linked child like has_pair gets no
    column of its own, since its parent hand_category_norm already
    represents the same concept -- see cfr_features.display_feature_keys)
    and one float column per action category (the *current* net's
    regret-matching probability for that action, given that row's own
    legal-action mask). raw_features is the same rows' full net-input
    vectors (row-aligned with df, i.e. same order/positions), kept around
    so _filtered_feature_importance can re-explain just the rows a filter
    selects without re-touching the reservoir. Everything else the UI does
    is just pandas filtering/grouping over df, computed once."""
    net, net_config, reservoir = _load_checkpoint(checkpoint_path)
    rng = np.random.default_rng(0)
    n = min(max_samples, len(reservoir))
    idx = rng.choice(len(reservoir), size=n, replace=False) if len(reservoir) else np.array([], dtype=np.int64)

    features = reservoir.features[idx]
    legal_masks = reservoir.legal_masks[idx]
    net.eval()
    with torch.no_grad():
        regrets = net(torch.from_numpy(features)).numpy()
    probs = cfr_actions.regret_matching_batch(regrets, legal_masks) if n else np.zeros((0, strategy.NUM_ACTION_CATEGORIES))

    displayed_keys = set(cfr_features.display_feature_keys(net_config.feature_keys))
    data = {}
    for i, key in enumerate(net_config.feature_keys):
        if key not in displayed_keys:
            continue
        labels = cfr_features.bucket_labels(key, features[:, i])
        data[_FEATURE_COL_PREFIX + key] = pd.Categorical(
            labels, categories=cfr_features.bucket_categories(key), ordered=True,
        )
    for i, label in enumerate(strategy.ACTION_CATEGORIES):
        data[_ACTION_COL_PREFIX + label] = probs[:, i]

    return pd.DataFrame(data), features


@st.cache_data(show_spinner="Ranking features by SHAP contribution for the current filters...")
def _filtered_feature_importance(
    checkpoint_path: str, max_samples: int, filters_key: tuple[tuple[str, tuple[str, ...]], ...],
) -> list[tuple[str, float]]:
    """One (feature_key, mean |SHAP|) pair per *displayed* feature (see
    cfr_features.display_feature_keys/fold_child_contributions), explained
    over only the reservoir rows matching `filters_key` -- a hashable
    (feature_key, kept_bucket_labels) form of the sidebar's current
    Filter-role selections, so this recomputes (and st.cache_data
    invalidates) whenever a filter is added, removed, or changed. The
    background reference is drawn from that *same* filtered pool, not the
    whole loaded sample: a feature the filter holds constant (e.g.
    num_overcards_norm within a preflop-only filter, where there's no
    board yet so it's always exactly 0) then has x_i - background_i = 0 for
    every interpolation point, so GradientExplainer attributes it ~0
    contribution structurally -- not just on average after centering (see
    cfr_networks._normalized_mean_abs_shap). Comparing against the whole
    reservoir instead would draw background_i from rows where the feature
    does vary (e.g. postflop hands), making x_i - background_i nonzero and
    noisy per explained row in a way row-mean centering doesn't fully
    cancel, letting an actually-constant-under-this-filter feature still
    read as importantly-contributing.

    Always returns exactly one entry per displayed feature -- including a
    0.0 entry for every feature when the filters match zero rows -- rather
    than the empty list mean_shap_contributions_for_samples would give back
    for an empty explain pool: _render_sidebar iterates this same list to
    decide which role/filter widgets to draw, so a feature dropping out of
    it would silently make that feature's own controls (and whatever
    filter it's set to) vanish from the sidebar."""
    net, net_config, _reservoir = _load_checkpoint(checkpoint_path)
    df, raw_features = _build_dataframe(checkpoint_path, max_samples)
    filters = {key: list(values) for key, values in filters_key}
    mask = _filter_mask(df, filters).to_numpy()
    filtered_features = raw_features[mask]
    contributions = cfr_networks.mean_shap_contributions_for_samples(
        net, filtered_features, filtered_features, net_config.feature_keys, np.random.default_rng(0),
    )
    folded = dict(cfr_features.fold_child_contributions(contributions))
    display_keys = cfr_features.display_feature_keys(net_config.feature_keys)
    return sorted(((key, folded.get(key, 0.0)) for key in display_keys), key=lambda kv: -kv[1])


def _feature_col(key: str) -> str:
    return _FEATURE_COL_PREFIX + key


def _observed_categories(df: pd.DataFrame, key: str) -> list[str]:
    """Every bucket label of `key` that actually shows up in `df` -- the
    sidebar only ever offers filter/observed values a person could actually
    select something for, not the feature's full theoretical value table."""
    return [c for c in cfr_features.bucket_categories(key) if c in set(df[_feature_col(key)])]


def _current_filters_from_session_state(display_keys: list[str], df: pd.DataFrame) -> dict[str, list[str]]:
    """Rebuilds the `filters` dict from whatever's already sitting in
    st.session_state for each feature's role/filter widgets -- so
    _filtered_feature_importance can be computed *before* _render_sidebar
    re-creates those same widgets later in this same rerun. Safe to read
    this early because Streamlit keeps a widget's session_state value
    across reruns keyed by its own `key=`, independent of when in the
    script that widget actually gets re-instantiated."""
    filters: dict[str, list[str]] = {}
    for key in display_keys:
        if st.session_state.get(f"role::{key}") != ROLE_FILTER:
            continue
        filters[key] = st.session_state.get(f"filter::{key}", _observed_categories(df, key))
    return filters


def _action_columns(collapsed: bool) -> list[str]:
    labels = _COLLAPSED_LABELS if collapsed else strategy.ACTION_CATEGORIES
    return [_ACTION_COL_PREFIX + label for label in labels]


def _with_action_view(df: pd.DataFrame, collapsed: bool) -> pd.DataFrame:
    """`df` with its action-probability columns either left as the native 9
    categories, or summed down to Fold/Call/Raise/All-In (mutually
    exclusive components of one distribution, so summing -- not averaging
    -- is what "the probability of any raise size" means)."""
    if not collapsed:
        return df
    out = df[[c for c in df.columns if not c.startswith(_ACTION_COL_PREFIX)]].copy()
    for collapsed_label in _COLLAPSED_LABELS:
        source_cols = [
            _ACTION_COL_PREFIX + strategy.ACTION_CATEGORIES[i]
            for i, group in _COLLAPSED_GROUP_OF_ACTION.items() if group == collapsed_label
        ]
        out[_ACTION_COL_PREFIX + collapsed_label] = df[source_cols].sum(axis=1)
    return out


def _render_sidebar(
    feature_order: list[tuple[str, float]], df: pd.DataFrame,
) -> tuple[dict[str, list[str]], list[str], list[str], bool]:
    """Returns (filters, group_split_keys, table_split_keys, collapsed).
    `filters` maps feature_key -> the bucket labels to keep for it."""
    st.sidebar.header("Features")
    st.sidebar.caption(
        "Ordered by mean |SHAP| contribution to the net's predictions, over whichever rows "
        "currently pass your Filter selections below (recomputed whenever a filter changes)."
    )

    filters: dict[str, list[str]] = {}
    group_split_keys: list[str] = []
    table_split_keys: list[str] = []

    for key, importance in feature_order:
        label = f"{cfr_features.feature_label(key)}  (SHAP {importance:.4f})"
        role = st.sidebar.selectbox(
            label, ROLES, key=f"role::{key}", help=cfr_features.feature_description(key),
        )
        if role == ROLE_FILTER:
            observed = _observed_categories(df, key)
            filters[key] = st.sidebar.multiselect("keep values", observed, default=observed, key=f"filter::{key}")
        elif role == ROLE_GROUP_SPLIT:
            group_split_keys.append(key)
        elif role == ROLE_TABLE_SPLIT:
            table_split_keys.append(key)

    if len(table_split_keys) > MAX_TABLE_SPLIT_FEATURES:
        kept, dropped = table_split_keys[:MAX_TABLE_SPLIT_FEATURES], table_split_keys[MAX_TABLE_SPLIT_FEATURES:]
        kept_labels = ", ".join(cfr_features.feature_label(k) for k in kept)
        dropped_labels = ", ".join(cfr_features.feature_label(k) for k in dropped)
        st.error(
            f"Only {MAX_TABLE_SPLIT_FEATURES} features can be used as table splits at once -- "
            f"using {kept_labels} (by SHAP rank) and ignoring {dropped_labels}. Change one "
            "of those features' roles to use a different pair."
        )
        table_split_keys = kept

    st.sidebar.divider()
    collapsed = st.sidebar.toggle("Collapse actions to Fold / Call / Raise / All-In", value=False)

    return filters, group_split_keys, table_split_keys, collapsed


def _filter_mask(df: pd.DataFrame, filters: dict[str, list[str]]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for key, values in filters.items():
        mask &= df[_feature_col(key)].isin(values)
    return mask


def _apply_filters(df: pd.DataFrame, filters: dict[str, list[str]]) -> pd.DataFrame:
    return df[_filter_mask(df, filters)]


def _render_active_filters(filters: dict[str, list[str]]) -> None:
    """A quick-remove control for the central section, just above the
    tables: every feature currently marked as a Filter shows up as one tag
    in a multiselect, and Streamlit renders each selected tag with its own
    small "x" -- clicking it off here has the same effect as switching that
    feature's role back to Unused in the sidebar, without having to go find
    it there (the sidebar list is SHAP-ordered, so a given feature isn't
    always where you last left it).

    Keyed on the *current* set of active filter keys (rather than one fixed
    key) so a change in which features are filters -- made either here or
    in the sidebar -- always starts this widget from a fresh, fully-checked
    default instead of risking a stale selection left over from a
    since-removed filter."""
    if not filters:
        return
    active_keys = list(filters.keys())
    widget_key = "active_filters::" + "|".join(sorted(active_keys))
    kept_keys = st.multiselect(
        "Active filters (click a tag's ✕ to turn that filter off)",
        options=active_keys, default=active_keys, format_func=cfr_features.feature_label,
        key=widget_key,
    )
    removed_keys = set(active_keys) - set(kept_keys)
    if removed_keys:
        for key in removed_keys:
            st.session_state[f"role::{key}"] = ROLE_UNUSED
        st.rerun()


def _render_table(group_df: pd.DataFrame, table_split_keys: list[str], collapsed: bool) -> None:
    action_cols = _action_columns(collapsed)
    view = _with_action_view(group_df, collapsed)

    if not table_split_keys:
        summary = view[action_cols].mean().to_frame("All rows").T
        counts = pd.DataFrame({"n": [len(view)]}, index=["All rows"])
    elif len(table_split_keys) == 1:
        idx_col = _feature_col(table_split_keys[0])
        row_label = cfr_features.feature_label(table_split_keys[0])
        summary = view.groupby(idx_col, observed=True)[action_cols].mean()
        counts = view.groupby(idx_col, observed=True).size().to_frame("n")
        summary.index.name = row_label
        counts.index.name = row_label
    else:
        row_col, col_col = _feature_col(table_split_keys[0]), _feature_col(table_split_keys[1])
        row_label = cfr_features.feature_label(table_split_keys[0])
        col_label = cfr_features.feature_label(table_split_keys[1])
        summary = pd.pivot_table(view, values=action_cols, index=row_col, columns=col_col, aggfunc="mean", observed=True)
        summary = summary.swaplevel(axis=1).sort_index(axis=1, level=0)
        counts = pd.pivot_table(view, values=action_cols[0], index=row_col, columns=col_col, aggfunc="count", observed=True)
        summary.index.name = row_label
        counts.index.name = row_label
        summary.columns = summary.columns.set_names(col_label, level=0)
        counts.columns.name = col_label

    st.dataframe((summary * 100).round(1).astype(str) + "%")
    with st.expander(f"Sample counts (total n={len(view):,})"):
        st.dataframe(counts)


def _render_group_heading(text: str, level: int) -> None:
    """level 0 uses st.subheader (matching this app's single-group-split
    behavior before nesting existed); each level below that drops one
    further markdown heading size, capped at h6, so a chain of several
    group splits still reads as a strict hierarchy instead of running out
    of distinct sizes."""
    if level == 0:
        st.subheader(text)
    else:
        st.markdown(f"{'#' * min(level + 3, 6)} {text}")


def _render_grouped(
    df: pd.DataFrame, group_split_keys: list[str], table_split_keys: list[str], collapsed: bool, level: int = 0,
) -> None:
    """One heading per observed value of group_split_keys[0], each nested
    under the previous one (see _render_group_heading) and recursed into
    for the remaining group_split_keys -- so with several group splits
    selected, a table sits under a chain of headings each naming just its
    own feature/value (e.g. Street = Flop, then nested under it Position =
    Late), rather than one flat heading repeating every key/value pair
    above every leaf table."""
    if not group_split_keys:
        _render_table(df, table_split_keys, collapsed)
        return

    key, *rest = group_split_keys
    col = _feature_col(key)
    label = cfr_features.feature_label(key)
    for value, group_df in df.groupby(col, observed=True):
        if len(group_df) == 0:
            continue
        _render_group_heading(f"{label} = {value}  (n={len(group_df):,})", level)
        _render_grouped(group_df, rest, table_split_keys, collapsed, level + 1)


def main() -> None:
    st.set_page_config(page_title="CFR Strategy Explorer", layout="wide")
    st.title("CFR Strategy Explorer")

    checkpoint_path = st.sidebar.text_input("Checkpoint path", value=_default_checkpoint_path())
    max_samples = st.sidebar.number_input(
        "Max reservoir samples to load", min_value=100, max_value=1_000_000,
        value=DEFAULT_MAX_SAMPLES, step=1000,
    )

    if not (os.path.exists(f"{checkpoint_path}.pt") and os.path.exists(f"{checkpoint_path}.json")):
        st.error(f"No checkpoint found at {checkpoint_path}.{{pt,json}}")
        st.stop()

    _net, net_config, _reservoir = _load_checkpoint(checkpoint_path)
    df, _raw_features = _build_dataframe(checkpoint_path, int(max_samples))
    if df.empty:
        st.warning("This checkpoint's reservoir is empty -- nothing to explore yet.")
        st.stop()

    st.caption(f"{len(df):,} reservoir samples loaded.")

    display_keys = cfr_features.display_feature_keys(net_config.feature_keys)
    current_filters = _current_filters_from_session_state(display_keys, df)
    # Rendered (and, on a tag removal, may st.rerun()) before _render_sidebar
    # below instantiates the matching role::key widgets -- session_state for
    # an already-instantiated widget can't be reassigned this same run.
    _render_active_filters(current_filters)

    filters_key = tuple(sorted((key, tuple(values)) for key, values in current_filters.items()))
    feature_importance = _filtered_feature_importance(checkpoint_path, int(max_samples), filters_key)

    filters, group_split_keys, table_split_keys, collapsed = _render_sidebar(feature_importance, df)
    filtered = _apply_filters(df, filters)

    if filtered.empty:
        st.warning("No reservoir samples match the current filters.")
        st.stop()

    _render_grouped(filtered, group_split_keys, table_split_keys, collapsed)


main()
