"""Interactive Streamlit app for turning a trained Single Deep CFR strategy
into a human-*implementable* one: a tree of "sub-strategies", each an
ordered, filtered slice of the reservoir with its own 1-2 "Split By"
features whose exact table+graph become its implementable rule, a
%SHAP-explained figure showing how much predictive signal that
simplification keeps, and its own further-nested child sub-strategies.

Every sub-strategy -- including the root one, covering the whole loaded
reservoir -- gets the exact same central controls: "Add filter" (for a
child, this *is* its claim condition -- see below), "Split By" (1-2
features, defaulting to inherit its parent's current pair, independently
overridable), "Add sub-strategy", and purely illustrative "Add table"/"Add
graph" additions. There's no separate global/sidebar version of any of
these -- the sidebar is just page-level settings (checkpoint path, sample
cap, the collapse toggle) plus a navigation tree (see _render_navigation)
of clickable links, one per sub-strategy, for jumping around the page
instead of scrolling.

Sub-strategies claim rows in priority order: each child sees only its
parent's rows minus whatever earlier siblings already claimed (via their
own filters), and a sub-strategy's actual "default behaviour" -- the
prominent table+graph -- is whatever's left of its own rows once every one
of its children has claimed its share. That's deliberately the same shape
as a real "if this situation, do X; elif that, do Y; else, do Z" rule list
a person could actually follow.

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

Exact Hole Hand (features.py's hole_hand_grid_x_norm, paired with
hole_hand_grid_y_norm which collapses into it -- see
cfr_features.display_feature_keys) is a partial exception: it's inherently
2D, a position in the classic 13x13 preflop starting-hand grid rather than
a single ordered scale, so it's excluded from "Add filter"/"Add table"
options, and picking it as Split By fills *both* slots by itself (see
_resolve_split_by) and only actually renders (as its own fixed set of
heatmaps, in place of the usual table+line-chart -- see
_hole_hand_grid_figures) once a sub-strategy's own rows are 100% preflop
(see _hole_hand_grid_split_by_available) -- masked to a negative sentinel
outside preflop (see features.extract_features), so it has nothing
meaningful to plot otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import uuid

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

import cfr_actions
import cfr_features
import cfr_networks
import cfr_reservoir
import strategy
# Named imports, not `import features` -- _build_dataframe already uses
# `features` as a local variable name for the raw reservoir feature matrix.
from features import HOLE_HAND_GRID_RANK_LABELS, HOLE_HAND_GRID_SIZE, MASKED, hole_hand_grid_label

DEFAULT_CHECKPOINT_PATH = os.path.join("cfr_runs", "checkpoint_latest")
DEFAULT_MAX_SAMPLES = 1_000_000

_FEATURE_COL_PREFIX = "feat::"
_ACTION_COL_PREFIX = "action::"

MAX_SPLIT_BY_FEATURES = 2

# Exact Hole Hand: the display-representative key (features.py links
# hole_hand_grid_y_norm to this one so display_feature_keys collapses them
# into a single entry in every dropdown -- see that FeatureSpec's own
# docstring) and its own second axis, plus the two raw (non-bucket-labeled)
# DataFrame columns _build_dataframe stores its actual values under -- see
# _hole_hand_grid_figures for why a generic bucket-label column wouldn't
# make sense for it the way it does for every other feature.
_HOLE_HAND_GRID_KEY = "hole_hand_grid_x_norm"
_HOLE_HAND_GRID_Y_KEY = "hole_hand_grid_y_norm"
_HOLE_HAND_GRID_RAW_X_COL = "raw::hole_hand_grid_x_norm"
_HOLE_HAND_GRID_RAW_Y_COL = "raw::hole_hand_grid_y_norm"

_COLLAPSED_LABELS = ("Fold", "Call", "Raise", "All-In")
_COLLAPSED_GROUP_OF_ACTION = {
    strategy.ACTION_FOLD: "Fold",
    strategy.ACTION_CALL: "Call",
    **{a: "Raise" for a in strategy.RAISE_ACTIONS},
    strategy.ACTION_ALLIN: "All-In",
}

# A fixed-order categorical palette (identity, never re-cycled/re-ranked by
# whatever's currently on screen) for the 4 collapsed action groups -- kept
# to just these first 4 slots since they're the ones validated for every
# adjacent pair in both light and dark modes.
_CATEGORICAL_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
# The 9 native action categories are really one ordered "aggression" scale
# (Fold -> Call -> Raise 25% -> ... -> All-In), so they read better as a
# single-hue ordinal ramp (light = passive, dark = aggressive) than as 9
# arbitrary categorical hues -- also sidesteps the palette only having 8
# categorical slots defined. Steps 250-650 off the validated sequential blue
# ramp (light mode's ordinal floor is step 250).
_ORDINAL_BLUE_9 = (
    "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
    "#256abf", "#1c5cab", "#184f95", "#104281",
)
# The full sequential ramp (magnitude: 0% -> 100%), used for every heatmap's
# colorscale -- each heatmap is its own independent panel (never two
# sequential fields overlaid at once), so reusing one hue across all of
# them is fine; the title text carries which action it is, not the color.
_SEQUENTIAL_BLUE_SCALE = [
    (0 / 12, "#cde2fb"), (1 / 12, "#b7d3f6"), (2 / 12, "#9ec5f4"), (3 / 12, "#86b6ef"),
    (4 / 12, "#6da7ec"), (5 / 12, "#5598e7"), (6 / 12, "#3987e5"), (7 / 12, "#2a78d6"),
    (8 / 12, "#256abf"), (9 / 12, "#1c5cab"), (10 / 12, "#184f95"), (11 / 12, "#104281"), (12 / 12, "#0d366b"),
]
_CHART_GRIDLINE_COLOR = "#e1e0d9"
_CHART_AXIS_COLOR = "#c3c2b7"
_CHART_MUTED_TEXT_COLOR = "#898781"
_CHART_PRIMARY_TEXT_COLOR = "#0b0b0b"


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
    cfr_features.bucket_categories; a linked child like hole_hand_grid_y_norm
    gets no column of its own, since its parent hole_hand_grid_x_norm already
    represents the same concept -- see cfr_features.display_feature_keys)
    and one float column per action category (the *current* net's
    regret-matching probability for that action, given that row's own
    legal-action mask). Exact Hole Hand (_HOLE_HAND_GRID_KEY) is the one
    exception: rather than a generic bucket-label column (its own value
    table is just 13 per-axis ranks, not the 169 combos it actually
    represents jointly with its second axis), it gets two plain raw float
    columns instead -- see _hole_hand_grid_figures, which reads them
    directly. raw_features is the same rows' full net-input vectors
    (row-aligned with df, i.e. same order/positions), kept around so
    _shap_importance_for_rows can re-explain just the rows a sub-strategy
    claims without re-touching the reservoir. Everything else the UI does
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
        if key not in displayed_keys or key == _HOLE_HAND_GRID_KEY:
            continue
        labels = cfr_features.bucket_labels(key, features[:, i])
        data[_FEATURE_COL_PREFIX + key] = pd.Categorical(
            labels, categories=cfr_features.bucket_categories(key), ordered=True,
        )
    for i, label in enumerate(strategy.ACTION_CATEGORIES):
        data[_ACTION_COL_PREFIX + label] = probs[:, i]

    if _HOLE_HAND_GRID_KEY in net_config.feature_keys and _HOLE_HAND_GRID_Y_KEY in net_config.feature_keys:
        x_idx = net_config.feature_keys.index(_HOLE_HAND_GRID_KEY)
        y_idx = net_config.feature_keys.index(_HOLE_HAND_GRID_Y_KEY)
        data[_HOLE_HAND_GRID_RAW_X_COL] = features[:, x_idx]
        data[_HOLE_HAND_GRID_RAW_Y_COL] = features[:, y_idx]

    return pd.DataFrame(data), features


def _row_digest(row_index: np.ndarray) -> str:
    """Cheap, stable hash of a set of row positions -- the actual
    st.cache_data key for _shap_importance_for_rows, since the row array
    itself (potentially large) is excluded from Streamlit's own hashing via
    its leading-underscore parameter name."""
    return hashlib.sha1(np.ascontiguousarray(row_index).tobytes()).hexdigest()


def _group_labels_for_rows(df: pd.DataFrame, row_index: np.ndarray, group_by_keys: tuple[str, ...]) -> np.ndarray | None:
    """One group-identifying value per row at `row_index`, defined by
    `group_by_keys` -- a parent sub-strategy's own *resolved* Split By
    pair (see _resolve_split_by) -- for _shap_importance_for_rows to center
    a sub-strategy's own SHAP contributions per parent-group instead of
    over its whole row set (see cfr_networks._normalized_mean_abs_shap).
    None if `group_by_keys` is empty (no parent grouping in effect, e.g.
    root, or a parent with no Split By chosen yet).

    Exact Hole Hand (_HOLE_HAND_GRID_KEY) is handled separately since it
    has no ordinary bucket-label column (see _build_dataframe): grouped by
    its own 13x13 grid cell instead, with every masked (postflop) row
    grouped together under one shared "masked" label -- a raw
    (x, y) < 0 pair is a sentinel, not a real grid position, so binning it
    into a grid cell (which would land it at cell (0, 0), colliding with a
    real AA reading) would be wrong."""
    if not group_by_keys:
        return None
    if group_by_keys == (_HOLE_HAND_GRID_KEY,):
        x = df[_HOLE_HAND_GRID_RAW_X_COL].to_numpy()
        y = df[_HOLE_HAND_GRID_RAW_Y_COL].to_numpy()
        size = HOLE_HAND_GRID_SIZE
        cols = np.rint(np.clip(x, 0.0, 1.0) * (size - 1)).astype(int)
        rows = np.rint(np.clip(y, 0.0, 1.0) * (size - 1)).astype(int)
        cell = rows * size + cols
        labels = np.where(x < 0.0, -1, cell)
        return labels[row_index]
    columns = [df[_feature_col(key)].to_numpy().astype(str) for key in group_by_keys]
    combined = columns[0] if len(columns) == 1 else np.char.add(np.char.add(columns[0], "|"), columns[1])
    return combined[row_index]


@st.cache_data(show_spinner="Ranking features by SHAP contribution for this sub-strategy...")
def _shap_importance_for_rows(
    checkpoint_path: str, max_samples: int, row_digest: str, _row_index: np.ndarray,
    group_by_keys: tuple[str, ...] = (),
) -> list[tuple[str, float]]:
    """Mean |SHAP| contribution per displayed feature (see
    cfr_networks.mean_shap_contributions_for_samples/cfr_features.
    fold_child_contributions/display_feature_keys), restricted to exactly
    the rows at `_row_index` -- a sub-strategy's own claimed rows, however
    they got there (its own filters, minus whatever earlier siblings
    already claimed -- not representable as a simple
    feature->kept-bucket-labels dict, since claim-order exclusion isn't
    "pin one key to one value"). Used both for a sub-strategy's own
    "Add ..." dropdown ranking and its %SHAP-explained figure. Always one
    entry per displayed feature, 0.0 for every feature when `_row_index` is
    empty, so a feature never silently drops out of a widget that iterates
    this list to build its options.

    `group_by_keys` (a parent sub-strategy's own resolved Split By pair --
    see _render_substrategy) makes this sub-strategy's own view assume that
    parent grouping is already "priced in": each row's contribution is
    centered against its own group's mean (see _group_labels_for_rows/
    cfr_networks._normalized_mean_abs_shap) instead of the whole row set's
    mean, so a feature the parent's grouping already fully explains (e.g.
    Exact Hole Hand fully determining Hole Suited) scores ~0 here too,
    rather than getting credited again for signal the parent's own
    grouping already accounts for."""
    net, net_config, _reservoir = _load_checkpoint(checkpoint_path)
    display_keys = cfr_features.display_feature_keys(net_config.feature_keys)
    if len(_row_index) == 0:
        return [(key, 0.0) for key in display_keys]
    df, raw_features = _build_dataframe(checkpoint_path, max_samples)
    selected = raw_features[_row_index]
    group_labels = _group_labels_for_rows(df, _row_index, group_by_keys)
    contributions = cfr_networks.mean_shap_contributions_for_samples(
        net, selected, selected, net_config.feature_keys, np.random.default_rng(0),
        explain_group_labels=group_labels,
    )
    folded = dict(cfr_features.fold_child_contributions(contributions))
    return sorted(((key, folded.get(key, 0.0)) for key in display_keys), key=lambda kv: -kv[1])


def _feature_col(key: str) -> str:
    return _FEATURE_COL_PREFIX + key


def _resolve_split_by(chosen_keys: list[str]) -> list[str]:
    """If Exact Hole Hand (_HOLE_HAND_GRID_KEY) is among `chosen_keys`, it
    alone fills both of Split By's slots -- it's already inherently 2D, so
    any other co-selected feature is dropped (callers render an st.caption
    explaining why when that happens). Otherwise just the first
    MAX_SPLIT_BY_FEATURES picks, in whatever order they were chosen."""
    if _HOLE_HAND_GRID_KEY in chosen_keys:
        return [_HOLE_HAND_GRID_KEY]
    return list(chosen_keys[:MAX_SPLIT_BY_FEATURES])


def _hole_hand_grid_available(df: pd.DataFrame) -> bool:
    """Whether `df` has at least one preflop (unmasked) Exact Hole Hand
    row -- used to decide whether to offer it in a sub-strategy's own "Add
    graph" dropdown (see _render_substrategy). False if the checkpoint
    wasn't trained with the feature at all (no raw column to check)."""
    if _HOLE_HAND_GRID_RAW_X_COL not in df.columns:
        return False
    return bool((df[_HOLE_HAND_GRID_RAW_X_COL] >= 0.0).any())


def _hole_hand_grid_split_by_available(df: pd.DataFrame) -> bool:
    """Whether *every* row of `df` is preflop (unmasked) Exact Hole Hand --
    stricter than _hole_hand_grid_available's `.any()` (used for "Add
    graph" eligibility, where a partial view is still fine to chart).
    Split By is meant to be exactly implementable, so picking Exact Hole
    Hand there only actually renders once a sub-strategy's own rows are
    100% preflop -- see _render_substrategy."""
    if _HOLE_HAND_GRID_RAW_X_COL not in df.columns or df.empty:
        return False
    return bool((df[_HOLE_HAND_GRID_RAW_X_COL] >= 0.0).all())


def _observed_categories(df: pd.DataFrame, key: str) -> list[str]:
    """Every bucket label of `key` that actually shows up in `df` -- the
    sidebar only ever offers filter/observed values a person could actually
    select something for, not the feature's full theoretical value table."""
    return [c for c in cfr_features.bucket_categories(key) if c in set(df[_feature_col(key)])]


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


def _chart_layout_kwargs(**extra) -> dict:
    """Layout options shared by every plotly figure this app draws: recessive
    hairline gridlines/axis, muted tick text, primary-ink title, transparent
    surface (so it sits on Streamlit's own background rather than fighting it
    with a second white card), and enough margin to avoid clipping labels."""
    return dict(
        font=dict(color=_CHART_MUTED_TEXT_COLOR),
        title_font=dict(color=_CHART_PRIMARY_TEXT_COLOR),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=48, b=40, l=10, r=10),
        **extra,
    )


def _line_chart_figure(filtered: pd.DataFrame, x_key: str, collapsed: bool) -> go.Figure:
    """A line per action category (the native 9-way aggression scale, or the
    4 collapsed groups if `collapsed`), mean action probability against every
    observed bucket of `x_key`, y axis pinned to the full 0-100% range so
    different features' charts stay visually comparable."""
    view = _with_action_view(filtered, collapsed)
    action_cols = _action_columns(collapsed)
    action_labels = _COLLAPSED_LABELS if collapsed else strategy.ACTION_CATEGORIES
    colors = _CATEGORICAL_COLORS if collapsed else _ORDINAL_BLUE_9
    x_col = _feature_col(x_key)
    order = _observed_categories(filtered, x_key)
    summary = view.groupby(x_col, observed=True)[action_cols].mean().reindex(order)

    fig = go.Figure()
    for label, col, color in zip(action_labels, action_cols, colors):
        fig.add_trace(go.Scatter(
            x=order, y=summary[col], name=label, mode="lines+markers",
            line=dict(width=2, color=color), marker=dict(size=8, color=color),
            hovertemplate=f"{label}: " + "%{y:.1%}<extra></extra>",
        ))
    fig.update_layout(
        **_chart_layout_kwargs(),
        title=f"{cfr_features.feature_label(x_key)} vs. action probability",
        xaxis=dict(
            title=cfr_features.feature_label(x_key), type="category",
            categoryorder="array", categoryarray=order,
            gridcolor=_CHART_GRIDLINE_COLOR, linecolor=_CHART_AXIS_COLOR,
        ),
        yaxis=dict(
            title="Action probability", range=[0, 1], tickformat=".0%",
            gridcolor=_CHART_GRIDLINE_COLOR, linecolor=_CHART_AXIS_COLOR,
        ),
        legend=dict(title="Action"), hovermode="x unified", height=420,
    )
    return fig


def _heatmap_figures(filtered: pd.DataFrame, x_key: str, y_key: str) -> list[go.Figure]:
    """One heatmap per collapsed action group (always the 4 simplified
    groups, regardless of the sidebar's collapse toggle -- a 9-way split
    wouldn't fit sensibly on a 2D grid meant to be read at a glance): mean
    action rate for every observed (x_key, y_key) bucket combination."""
    view = _with_action_view(filtered, collapsed=True)
    x_col, y_col = _feature_col(x_key), _feature_col(y_key)
    x_order = _observed_categories(filtered, x_key)
    y_order = _observed_categories(filtered, y_key)
    x_label, y_label = cfr_features.feature_label(x_key), cfr_features.feature_label(y_key)

    figures = []
    for action_label in _COLLAPSED_LABELS:
        action_col = _ACTION_COL_PREFIX + action_label
        pivot = view.pivot_table(values=action_col, index=y_col, columns=x_col, aggfunc="mean", observed=True)
        pivot = pivot.reindex(index=y_order, columns=x_order)
        z = pivot.to_numpy(dtype=float)
        text = np.full(z.shape, "", dtype=object)
        text[~np.isnan(z)] = [f"{v:.0%}" for v in z[~np.isnan(z)]]

        fig = go.Figure(data=go.Heatmap(
            z=z, x=x_order, y=y_order, text=text, texttemplate="%{text}",
            zmin=0, zmax=1, colorscale=_SEQUENTIAL_BLUE_SCALE,
            colorbar=dict(title="Rate", tickformat=".0%"),
            hovertemplate=f"{x_label}: %{{x}}<br>{y_label}: %{{y}}<br>{action_label}: %{{z:.1%}}<extra></extra>",
        ))
        fig.update_layout(
            **_chart_layout_kwargs(),
            title=f"{action_label} rate",
            xaxis=dict(title=x_label, type="category", categoryorder="array", categoryarray=x_order),
            yaxis=dict(title=y_label, type="category", categoryorder="array", categoryarray=y_order),
            height=380,
        )
        figures.append(fig)
    return figures


def _hole_hand_grid_figures(df: pd.DataFrame) -> list[go.Figure]:
    """One heatmap per collapsed action group (see _heatmap_figures for why
    always these 4, never the native 9), mean action rate over every one of
    the 169 exact starting hands, restricted to `df`'s own preflop
    (unmasked) rows -- postflop rows are dropped rather than pooled in,
    since the mask sentinel (features.MASKED) is a real, if
    negative, number that would otherwise corrupt the AA cell's own
    statistics. Empty list if `df` has no preflop rows (see
    _hole_hand_grid_available, checked by callers before rendering a
    heading for this). Unlike _heatmap_figures, every one of the 169 cells
    is always drawn and labeled with its combo code (see
    features.hole_hand_grid_label) whether or not any sample landed there
    -- this is a small, fixed, familiar chart, not an arbitrary feature
    pairing, so it reads better shown in full than restricted to whatever
    happens to be observed."""
    preflop_df = df[df[_HOLE_HAND_GRID_RAW_X_COL] >= 0.0]
    if preflop_df.empty:
        return []
    x = preflop_df[_HOLE_HAND_GRID_RAW_X_COL].to_numpy()
    y = preflop_df[_HOLE_HAND_GRID_RAW_Y_COL].to_numpy()

    size = HOLE_HAND_GRID_SIZE
    rows = np.rint(y * (size - 1)).astype(int)
    cols = np.rint(x * (size - 1)).astype(int)
    combo_text = np.array([[hole_hand_grid_label(r, c) for c in range(size)] for r in range(size)])

    # go.Heatmap draws z's first row at the bottom -- flip vertically (and
    # reverse the y tick labels to match) so grid row 0 (Ace) ends up at
    # the top and column 0 (also Ace) stays on the left, matching every
    # real range-chart tool: Ace/Ace in the top-left corner, deuce/deuce in
    # the bottom-right, suited hands (row < col) upper-right of that
    # diagonal, offsuit (row > col) lower-left.
    combo_text = combo_text[::-1]
    y_labels = list(reversed(HOLE_HAND_GRID_RANK_LABELS))
    x_labels = list(HOLE_HAND_GRID_RANK_LABELS)

    view = _with_action_view(preflop_df, collapsed=True)
    figures = []
    for action_label in _COLLAPSED_LABELS:
        action_col = _ACTION_COL_PREFIX + action_label
        vals = view[action_col].to_numpy()
        totals = np.zeros((size, size))
        counts = np.zeros((size, size))
        np.add.at(totals, (rows, cols), vals)
        np.add.at(counts, (rows, cols), 1)
        z = np.divide(totals, counts, out=np.full((size, size), np.nan), where=counts > 0)
        z = z[::-1]

        fig = go.Figure(data=go.Heatmap(
            z=z, x=x_labels, y=y_labels, text=combo_text, texttemplate="%{text}",
            zmin=0, zmax=1, colorscale=_SEQUENTIAL_BLUE_SCALE,
            colorbar=dict(title="Rate", tickformat=".0%"),
            hovertemplate="%{text}<br>" + f"{action_label}: " + "%{z:.1%}<extra></extra>",
        ))
        fig.update_layout(
            **_chart_layout_kwargs(),
            title=f"{action_label} rate",
            xaxis=dict(type="category", categoryorder="array", categoryarray=x_labels),
            yaxis=dict(type="category", categoryorder="array", categoryarray=y_labels),
            height=520,
        )
        figures.append(fig)
    return figures


def _render_graphs(
    filtered: pd.DataFrame, graph_keys: list[str], display_keys: list[str], collapsed: bool, key_prefix: str = "",
    level: int = 0,
) -> None:
    """One line chart per feature picked via a sub-strategy's own local
    "Add graph" control (see _render_substrategy), each paired with a
    multiselect of other features to "cross" it with -- every feature
    picked there adds its own row of 4 heatmaps (this graphed feature as
    the x axis, the picked feature as the y axis, one heatmap per
    simplified action rate). Exact Hole Hand (_HOLE_HAND_GRID_KEY) is the
    one exception: inherently 2D already, it renders its own fixed set of
    heatmaps in place of the line chart, with no "cross" multiselect (and
    is never itself offered as something else's cross target -- it has no
    ordinary bucket-label column to pivot on, see _build_dataframe).

    `key_prefix` disambiguates one call from another when _render_substrategy
    calls this once per sub-strategy node (so every node gets its own,
    independently-computed set of graphs over just its own rows) --
    without it, two nodes' widgets for the same graphed feature would
    collide on the same Streamlit widget key. `level` should already be
    one past whatever sub-strategy heading this call's graphs belong to --
    the "Graphs" heading here renders at exactly that level
    (_heading_at_level), so it always reads as a subsection of that
    sub-strategy rather than a sibling of equal weight."""
    if not graph_keys:
        return
    _heading_at_level("Graphs", level)

    for key in graph_keys:
        if key == _HOLE_HAND_GRID_KEY:
            st.markdown(f"**{cfr_features.feature_label(key)}**")
            figures = _hole_hand_grid_figures(filtered)
            if not figures:
                st.caption("No preflop rows in the current view.")
                continue
            heat_cols = st.columns(2)
            for i, fig in enumerate(figures):
                with heat_cols[i % 2]:
                    st.plotly_chart(fig, key=f"{key_prefix}graph_chart::{key}::{i}")
            continue

        other_keys = [k for k in display_keys if k != key and k != _HOLE_HAND_GRID_KEY]
        col_controls, col_chart = st.columns([1, 3])
        with col_controls:
            st.markdown(f"**{cfr_features.feature_label(key)}**")
            cross_keys = st.multiselect(
                "Cross with other features (adds heatmaps below)",
                options=other_keys, format_func=cfr_features.feature_label, key=f"{key_prefix}graph_cross::{key}",
            )
        with col_chart:
            st.plotly_chart(_line_chart_figure(filtered, key, collapsed), key=f"{key_prefix}graph_chart::{key}")

        for cross_key in cross_keys:
            st.caption(f"{cfr_features.feature_label(key)} × {cfr_features.feature_label(cross_key)}")
            # 2 per row rather than a cramped 4-across: each heatmap's own
            # per-cell % labels need real width to stay legible instead of
            # overlapping their neighbors.
            heat_cols = st.columns(2)
            for i, heatmap_fig in enumerate(_heatmap_figures(filtered, key, cross_key)):
                with heat_cols[i % 2]:
                    st.plotly_chart(heatmap_fig, key=f"{key_prefix}graph_heat::{key}::{cross_key}::{i}")


def _filter_mask(df: pd.DataFrame, filters: dict[str, list[str]]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for key, values in filters.items():
        mask &= df[_feature_col(key)].isin(values)
    return mask


def _apply_filters(df: pd.DataFrame, filters: dict[str, list[str]]) -> pd.DataFrame:
    return df[_filter_mask(df, filters)]


def _render_table(group_df: pd.DataFrame, split_by_keys: list[str], collapsed: bool) -> None:
    action_cols = _action_columns(collapsed)
    view = _with_action_view(group_df, collapsed)

    if not split_by_keys:
        summary = view[action_cols].mean().to_frame("All rows").T
        counts = pd.DataFrame({"n": [len(view)]}, index=["All rows"])
    elif len(split_by_keys) == 1:
        idx_col = _feature_col(split_by_keys[0])
        row_label = cfr_features.feature_label(split_by_keys[0])
        summary = view.groupby(idx_col, observed=True)[action_cols].mean()
        counts = view.groupby(idx_col, observed=True).size().to_frame("n")
        summary.index.name = row_label
        counts.index.name = row_label
    else:
        row_col, col_col = _feature_col(split_by_keys[0]), _feature_col(split_by_keys[1])
        row_label = cfr_features.feature_label(split_by_keys[0])
        col_label = cfr_features.feature_label(split_by_keys[1])
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


def _anchor_id(key_prefix: str) -> str:
    """A stable, HTML-safe in-page anchor id for the sub-strategy node at
    `key_prefix` (e.g. "root::" -> "root",
    "root::substrategy_ab12cd34::" -> "root-substrategy_ab12cd34") -- used
    by _render_substrategy (to mark each node's own heading) and
    _render_navigation (to link to it). Safe to splice into an
    unsafe_allow_html anchor tag since `key_prefix` is always built from
    this app's own code-controlled strings (the literal "root::" plus
    uuid4 hex child ids -- see _add_substrategy), never user input."""
    return key_prefix.rstrip(":").replace("::", "-")


def _heading_at_level(text: str, level: int) -> None:
    """level 0 uses st.subheader (matching this app's single-group-split
    behavior before nesting existed); each level below that drops one
    further markdown heading size, capped at h6, so a chain of several
    nested headings still reads as a strict hierarchy instead of running
    out of distinct sizes."""
    if level == 0:
        st.subheader(text)
    else:
        st.markdown(f"{'#' * min(level + 3, 6)} {text}")


_SUBSTRATEGY_CHILDREN_STATE_KEY = "substrategy_children"


def _substrategy_children(key_prefix: str) -> list[str]:
    """Ordered child sub-strategy ids currently nested directly under the
    node at `key_prefix`, persisted in st.session_state -- unlike the old
    automatic one-heading-per-observed-value nesting, which sub-strategies
    exist and in what priority order is now explicit, user-managed state
    (added one at a time via "Add sub-strategy"), not something derivable
    from the data alone."""
    return st.session_state.setdefault(_SUBSTRATEGY_CHILDREN_STATE_KEY, {}).setdefault(key_prefix, [])


def _add_substrategy(key_prefix: str) -> None:
    _substrategy_children(key_prefix).append(uuid.uuid4().hex[:8])


def _remove_substrategy(parent_key_prefix: str, child_id: str) -> None:
    # Deliberately doesn't try to recursively purge that child's own
    # descendant session_state/widget keys -- orphaned entries are inert
    # (never read again once nothing points at them) and the rest of this
    # app already doesn't bother with that kind of cleanup either (e.g. a
    # role switched away from Filter leaves its old filter::key values
    # sitting in session_state too).
    children = _substrategy_children(parent_key_prefix)
    if child_id in children:
        children.remove(child_id)


def _move_substrategy(parent_key_prefix: str, child_id: str, delta: int) -> None:
    children = _substrategy_children(parent_key_prefix)
    i = children.index(child_id)
    j = i + delta
    if 0 <= j < len(children):
        children[i], children[j] = children[j], children[i]


def _render_substrategy(
    incoming_df: pd.DataFrame, key_prefix: str, checkpoint_path: str, max_samples: int,
    display_keys: list[str], collapsed: bool, level: int,
    parent_key_prefix: str | None = None, child_id: str | None = None,
) -> pd.Index:
    """Renders one sub-strategy node -- root (`parent_key_prefix` is None,
    `incoming_df` the whole loaded reservoir, heading "Overall Strategy")
    or a child added via "Add sub-strategy" (its own local filters *are*
    its claim condition -- see the module docstring) -- and every one of
    its own further-nested children, in priority order. Every node gets
    identical controls; root differs only in having no Remove/Move
    buttons, a plain "Add filter" label instead of "Claim condition (Add
    filter)", and a prominent st.header instead of a level-sized
    _heading_at_level heading. Returns `incoming_df`'s row index this node
    claimed (after its own local filters, before any of its own children's
    further claims) so the caller can subtract it from what the *next*
    sibling sees, and from its own default-behaviour leftover once every
    child of *this* node has had a turn.

    Render order: this node's own local filters (define what it claims, so
    `node_df` here is its *full* claimed scope) -> anchor + heading (+
    remove/move controls for a child) -> "Add sub-strategy" + every
    existing child, recursed in order, each one seeing this node's rows
    minus every earlier sibling's own claim -> this node's actual "default
    behaviour" (`node_df` minus every child's own claim) -- Split By
    widget (own local widget, defaulting to inherit whatever its parent's
    *current* Split By pair is via a direct st.session_state read, since a
    child renders -- and needs its own default -- before its parent's own
    Split By widget runs this same script pass; root has no parent, so it
    just defaults to nothing), the prominent table+graph, and the
    %SHAP-explained figure, all computed over that leftover, since the
    prominent block *is* the default/else branch of the rule list -- then
    purely illustrative "Add table"/"Add graph" additions, also over that
    same leftover."""
    is_root = parent_key_prefix is None
    own_level = 0 if is_root else level

    # This node's own SHAP views (both calls below) assume its *parent's*
    # own Split By grouping is already "priced in" -- see
    # _group_labels_for_rows/cfr_networks._normalized_mean_abs_shap. Read
    # early (a plain session_state lookup, safe regardless of widget
    # render order) since it's also `chosen_split_by`'s own inherited
    # default further down.
    parent_split_by = [] if is_root else st.session_state.get(f"{parent_key_prefix}split_by", [])
    parent_group_by_keys = tuple(_resolve_split_by(parent_split_by))

    row_index = incoming_df.index.to_numpy()
    importance = _shap_importance_for_rows(
        checkpoint_path, max_samples, _row_digest(row_index), row_index, group_by_keys=parent_group_by_keys,
    )
    non_graph_options = [k for k, _ in importance if k != _HOLE_HAND_GRID_KEY]
    importance_by_key = dict(importance)

    def _claim_option_label(key: str) -> str:
        return f"{cfr_features.feature_label(key)}  (SHAP {importance_by_key[key]:.4f})"

    if not is_root:
        st.divider()
    filter_label = "Add filter" if is_root else "Claim condition (Add filter)"
    local_filter_keys = st.multiselect(
        filter_label, options=non_graph_options, format_func=_claim_option_label,
        key=f"{key_prefix}claim_filter_keys",
    )
    local_filters: dict[str, list[str]] = {}
    for filter_key in local_filter_keys:
        observed = _observed_categories(incoming_df, filter_key)
        local_filters[filter_key] = st.multiselect(
            f"{cfr_features.feature_label(filter_key)} -- keep values", options=observed, default=observed,
            key=f"{key_prefix}claim_filter_values::{filter_key}",
        )
    node_df = _apply_filters(incoming_df, local_filters)

    claim_desc = ", ".join(
        f"{cfr_features.feature_label(k)} = {', '.join(v)}" for k, v in local_filters.items()
    )
    st.markdown(f'<a name="{_anchor_id(key_prefix)}"></a>', unsafe_allow_html=True)
    if is_root:
        heading = f"Overall Strategy ({claim_desc})" if claim_desc else "Overall Strategy"
        st.header(f"{heading}  (n={len(node_df):,})")
    else:
        heading = claim_desc or "Sub-strategy (no filter set yet -- claims everything its parent hasn't)"
        col_heading, col_up, col_down, col_remove = st.columns([10, 1, 1, 1])
        with col_heading:
            _heading_at_level(f"{heading}  (n={len(node_df):,})", level)
        with col_up:
            st.button(
                "↑", key=f"{key_prefix}move_up", help="Move earlier (higher priority)",
                on_click=_move_substrategy, args=(parent_key_prefix, child_id, -1),
            )
        with col_down:
            st.button(
                "↓", key=f"{key_prefix}move_down", help="Move later (lower priority)",
                on_click=_move_substrategy, args=(parent_key_prefix, child_id, 1),
            )
        with col_remove:
            st.button(
                "✕", key=f"{key_prefix}remove", help="Remove this sub-strategy",
                on_click=_remove_substrategy, args=(parent_key_prefix, child_id),
            )

    if node_df.empty:
        st.warning("No rows match this sub-strategy's own claim filters.")
        return node_df.index

    st.button(
        "Add sub-strategy", key=f"{key_prefix}add_substrategy",
        on_click=_add_substrategy, args=(key_prefix,),
    )

    # Root's own heading doesn't consume a level (it has none), so its
    # direct children start at level 0, same as root's own `own_level`;
    # every deeper generation increments from its own parent's heading
    # level, matching _heading_at_level's tiering.
    child_level = 0 if is_root else own_level + 1
    remaining_df = node_df
    for this_child_id in list(_substrategy_children(key_prefix)):
        child_key_prefix = f"{key_prefix}substrategy_{this_child_id}::"
        claimed_index = _render_substrategy(
            remaining_df, child_key_prefix, checkpoint_path, max_samples,
            display_keys, collapsed, child_level,
            parent_key_prefix=key_prefix, child_id=this_child_id,
        )
        remaining_df = remaining_df.drop(claimed_index)

    default_df = remaining_df
    default_row_index = default_df.index.to_numpy()
    default_importance = _shap_importance_for_rows(
        checkpoint_path, max_samples, _row_digest(default_row_index), default_row_index,
        group_by_keys=parent_group_by_keys,
    )
    importance_by_key = dict(default_importance)

    def _default_option_label(key: str) -> str:
        return f"{cfr_features.feature_label(key)}  (SHAP {importance_by_key[key]:.4f})"

    split_by_options = [k for k, _ in default_importance]
    # A child defaults to inherit whatever its parent's *current* Split By
    # pair is (`parent_split_by`, read above) -- the parent's own Split By
    # widget (below, in the parent's own call to this function) hasn't run
    # yet this script pass by the time a child needs its default (children
    # render, and claim their own rows, before their parent's own Split By
    # widget does). Root has no parent to inherit from.
    chosen_split_by = st.multiselect(
        "Split By (this sub-strategy's own 1-2 implementable features)",
        options=split_by_options, default=[k for k in parent_split_by if k in split_by_options],
        format_func=_default_option_label, max_selections=MAX_SPLIT_BY_FEATURES,
        key=f"{key_prefix}split_by",
    )
    resolved_split_by = _resolve_split_by(chosen_split_by)
    if resolved_split_by != chosen_split_by:
        dropped = [k for k in chosen_split_by if k not in resolved_split_by]
        st.caption(
            f"Exact Hole Hand is inherently 2D and fills both Split By slots by itself -- "
            f"ignoring {', '.join(cfr_features.feature_label(k) for k in dropped)}."
        )

    st.markdown("**Default behaviour**" if not is_root else "**Overall default behaviour**")
    if not default_df.empty:
        if resolved_split_by == [_HOLE_HAND_GRID_KEY]:
            if _hole_hand_grid_split_by_available(default_df):
                figures = _hole_hand_grid_figures(default_df)
                heat_cols = st.columns(2)
                for i, fig in enumerate(figures):
                    with heat_cols[i % 2]:
                        st.plotly_chart(fig, key=f"{key_prefix}splitby_grid::{i}")
            else:
                postflop_mask = default_df[_HOLE_HAND_GRID_RAW_X_COL] < 0.0 if _HOLE_HAND_GRID_RAW_X_COL in default_df.columns else None
                postflop_frac = float(postflop_mask.mean()) if postflop_mask is not None else 1.0
                st.warning(
                    "Exact Hole Hand needs this sub-strategy's default-behaviour rows to be 100% "
                    f"preflop -- currently {postflop_frac:.0%} postflop. Add a Preflop claim filter "
                    "here to use it."
                )
        else:
            _render_table(default_df, resolved_split_by, collapsed)
            if resolved_split_by:
                if len(resolved_split_by) == 1:
                    st.plotly_chart(
                        _line_chart_figure(default_df, resolved_split_by[0], collapsed),
                        key=f"{key_prefix}splitby_chart",
                    )
                else:
                    heat_cols = st.columns(2)
                    figs = _heatmap_figures(default_df, resolved_split_by[0], resolved_split_by[1])
                    for i, fig in enumerate(figs):
                        with heat_cols[i % 2]:
                            st.plotly_chart(fig, key=f"{key_prefix}splitby_heat::{i}")

        if resolved_split_by:
            total = sum(v for _, v in default_importance)
            chosen_total = sum(v for k, v in default_importance if k in resolved_split_by)
            pct = (chosen_total / total * 100) if total > 0 else 0.0
            st.metric("SHAP explained by Split By features", f"{pct:.0f}%")
        else:
            st.caption("Pick 1-2 Split By features above to define this sub-strategy's implementable rule.")

    st.caption("Additional (illustrative) analysis")
    illustrative_options = [k for k in split_by_options if k != _HOLE_HAND_GRID_KEY]
    graph_options = split_by_options if _hole_hand_grid_available(default_df) else illustrative_options
    col_graph, col_table = st.columns(2)
    with col_graph:
        extra_graph_keys = st.multiselect(
            "Add graph", options=graph_options, format_func=_default_option_label, key=f"{key_prefix}extra_graph",
        )
    with col_table:
        extra_table_keys = st.multiselect(
            "Add table", options=illustrative_options, format_func=_default_option_label, key=f"{key_prefix}extra_table",
        )
    for table_key in extra_table_keys:
        st.caption(f"Extra table: {cfr_features.feature_label(table_key)}")
        _render_table(default_df, [table_key], collapsed)
    _render_graphs(default_df, extra_graph_keys, display_keys, collapsed, f"{key_prefix}extra::", own_level + 1)

    return node_df.index


def _shorthand_description(key_prefix: str) -> str:
    """The same claim-condition text this node's own heading shows (see
    _render_substrategy's `claim_desc`), rebuilt straight out of
    st.session_state instead of threaded through as a return value --
    _render_navigation runs once, after the whole sub-strategy tree has
    already rendered this script pass, so every node's own filter widgets'
    session_state is already fresh by the time this reads it."""
    filter_keys = st.session_state.get(f"{key_prefix}claim_filter_keys", [])
    parts = [
        f"{cfr_features.feature_label(key)} = {', '.join(st.session_state.get(f'{key_prefix}claim_filter_values::{key}', []))}"
        for key in filter_keys
    ]
    return ", ".join(parts)


def _truncate(text: str, limit: int = 40) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _nav_label(key_prefix: str, is_root: bool) -> str:
    desc = _truncate(_shorthand_description(key_prefix))
    if is_root:
        return f"Overall Strategy ({desc})" if desc else "Overall Strategy"
    return desc or "Sub-strategy (unclaimed)"


def _navigation_lines(key_prefix: str, is_root: bool, depth: int) -> list[str]:
    """One markdown link-list line for the node at `key_prefix`, plus one
    more (indented one level deeper) for each of its own children,
    recursively -- flattened into a single list rather than nested Python
    lists so the caller can join them into one markdown string, since
    st.markdown only renders a nested bullet list correctly when the whole
    thing is one call sharing consistent indentation, not several separate
    calls."""
    lines = [f"{'  ' * depth}- [{_nav_label(key_prefix, is_root)}](#{_anchor_id(key_prefix)})"]
    for child_id in _substrategy_children(key_prefix):
        lines.extend(_navigation_lines(f"{key_prefix}substrategy_{child_id}::", False, depth + 1))
    return lines


def _render_navigation() -> None:
    """The sidebar's nested, clickable shorthand outline of the current
    sub-strategy tree -- rendered after (in code order) _render_substrategy
    has already drawn the whole tree in the main body this same script
    pass, even though it visually appears in the sidebar, so every node's
    anchor already exists on the page and every node's own session_state
    (for its shorthand label) is already fresh."""
    st.sidebar.header("Navigation")
    st.sidebar.caption("Jump to any sub-strategy.")
    st.sidebar.markdown("\n".join(_navigation_lines("root::", True, 0)))


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

    st.sidebar.divider()
    collapsed = st.sidebar.toggle("Collapse actions to Fold / Call / Raise / All-In", value=False)

    # Root is just another sub-strategy node -- the whole loaded reservoir,
    # with its own local "Add filter"/"Split By" controls exactly like any
    # child (see _render_substrategy's module docstring).
    _render_substrategy(df, "root::", checkpoint_path, int(max_samples), display_keys, collapsed, 0)

    # Rendered after the tree above so it can read each node's own
    # session_state (filter picks, child list) freshly populated by the
    # widgets that just ran -- see _render_navigation.
    _render_navigation()


main()
