"""Interactive Streamlit app for turning a trained Single Deep CFR strategy
into a human-*implementable* one: a tree of "sub-strategies", each an
ordered, filtered slice of the reservoir with its own 1-2 "Split By"
features whose exact table+graph become its implementable rule, a
"Decision variance explained by Split By features on claimed samples"
metric showing how much predictive signal that simplification keeps, and
its own further-nested child sub-strategies.

Every sub-strategy -- including the root one, covering the whole loaded
reservoir -- gets the exact same central controls: "Add filter" (for a
child, this *is* its claim condition -- see below), "Split By" (1-2
features, defaulting to inherit its parent's current pair, independently
overridable), "Add sub-strategy", and purely illustrative "Add table"/"Add
graph" additions. There's no separate global/sidebar version of any of
these.

Only *one* sub-strategy's own content is ever on screen at a time (see
_render_substrategy) -- the sidebar's nested navigation tree (see
_render_navigation) doubles as a switcher: clicking a node there re-renders
the central column to show that node instead of scrolling to it. A node's
own page also lists its direct children as quick-jump buttons, and "Add
sub-strategy" jumps straight to the new child, so drilling down never
requires the sidebar; "← Back to parent" is the way back up.

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
import json
import os
import re
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
DEFAULT_MAX_SAMPLES = 2_000_000
# A big loaded reservoir pool is what gives an infrequent spot enough of
# its own rows to read a meaningful (i.e. not leave-one-out-crushed --
# see _decision_variance_explained) Decision variance explained figure;
# capping *analysis* per spot separately (this default), rather than
# capping the pool itself, keeps a common/high-volume spot's own table,
# chart, metrics, and every ranking/button fast regardless of how large
# that pool is loaded, without also making a rare spot's own comparatively
# tiny row count any smaller than it already naturally is.
DEFAULT_MAX_EVAL_SAMPLES = 100_000

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
def _build_dataframe(checkpoint_path: str, max_samples: int) -> pd.DataFrame:
    """One row per (subsampled) reservoir entry -- one Categorical column
    per *displayed* feature (its bucket label, in the feature's own
    natural value order -- see cfr_features.bucket_categories; a linked
    child like hole_hand_grid_y_norm gets no column of its own, since its
    parent hole_hand_grid_x_norm already represents the same concept --
    see cfr_features.display_feature_keys) and one float column per action
    category (the *current* net's regret-matching probability for that
    action, given that row's own legal-action mask). Exact Hole Hand
    (_HOLE_HAND_GRID_KEY) is the one exception: rather than a generic
    bucket-label column (its own value table is just 13 per-axis ranks,
    not the 169 combos it actually represents jointly with its second
    axis), it gets two plain raw float columns instead -- see
    _hole_hand_grid_figures, which reads them directly. Everything else
    the UI does -- every _decision_variance_explained/
    _decision_variance_by_key call included -- is just pandas
    filtering/grouping over this dataframe's own already-computed
    columns, computed once."""
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

    return pd.DataFrame(data)


def _group_labels_for_rows(df: pd.DataFrame, row_index: np.ndarray, group_by_keys: tuple[str, ...]) -> np.ndarray | None:
    """One group-identifying value per row at `row_index`, defined by
    `group_by_keys` -- either a parent sub-strategy's own *resolved* Split
    By pair (see _resolve_split_by), for _decision_variance_explained to
    center a sub-strategy's own residual per parent-group instead of over
    its whole row set, or the grouping whose own variance-explained is
    being measured in the first place (see _decision_variance_explained/
    _decision_variance_by_key) -- the latter can hold more than 2 keys at
    once (e.g. this node's own 2-feature Split By plus one more candidate
    being scored against it -- see _decision_variance_by_key's own
    `pair_with`), even though Split By itself never holds more than
    MAX_SPLIT_BY_FEATURES: that extra key still names a real, well-defined
    joint grouping to measure, it just isn't one a person could adopt as
    their own Split By pick directly. None if `group_by_keys` is empty (no
    grouping in effect, e.g. root with no parent, or a parent with no
    Split By chosen yet).

    Exact Hole Hand (_HOLE_HAND_GRID_KEY) is handled separately since it
    has no ordinary bucket-label column (see _build_dataframe): grouped by
    its own 13x13 grid cell instead (see _hole_hand_grid_cell_labels), with
    every masked (postflop) row grouped together under one shared "masked"
    label -- a raw (x, y) < 0 pair is a sentinel, not a real grid position,
    so binning it into a grid cell (which would land it at cell (0, 0),
    colliding with a real AA reading) would be wrong. It combines with
    other keys the same way any of them combine with each other -- e.g.
    this node's own current Split By being Exact Hole Hand alone (which
    fills both of Split By's own slots by itself -- see _resolve_split_by)
    doesn't stop a *candidate* feature from being scored jointly against
    it here: (grid cell, candidate's own bucket label) is just as
    well-defined a joint grouping as any other pair, even though a person
    could never adopt it as their own Split By pick directly (Split By's
    own 2-slot cap is a UI/memorability constraint on people, not a limit
    on what this function can measure)."""
    if not group_by_keys:
        return None
    columns = [
        _hole_hand_grid_cell_labels(df, row_index) if key == _HOLE_HAND_GRID_KEY
        # .iloc[row_index] *before* .to_numpy() -- df is often the whole
        # loaded reservoir (root_df) while row_index is a much smaller
        # subset (see _hole_hand_grid_cell_labels's own docstring on this
        # same point). A categorical column's .to_numpy() materializes a
        # full array of Python str objects for every one of df's own rows;
        # doing that over the whole column and only then indexing down
        # throws almost all of that work away, and scales with the full
        # reservoir size instead of len(row_index) -- indexing the
        # (cheap, codes-array-backed) Series first converts only the rows
        # actually needed.
        else df[_feature_col(key)].iloc[row_index].to_numpy().astype(str)
        for key in group_by_keys
    ]
    combined = columns[0]
    for column in columns[1:]:
        combined = np.char.add(np.char.add(combined, "|"), column)
    return combined


def _hole_hand_grid_cell_labels(df: pd.DataFrame, row_index: np.ndarray) -> np.ndarray:
    """One 13x13-grid-cell label per row at `row_index` -- "-1" for every
    masked (postflop) row, sharing one label rather than colliding with a
    real AA (cell (0, 0)) reading -- see _group_labels_for_rows. Indexed
    down to `row_index` *before* any per-row math below, not after -- `df`
    can be the whole loaded reservoir (root_df), while `row_index` is often
    a much smaller claimed/capped subset of it (see _capped_for_eval);
    doing the math over every row in `df` only to immediately throw most
    of it away would scale with the full reservoir size instead of with
    however many rows are actually being grouped here."""
    x = df[_HOLE_HAND_GRID_RAW_X_COL].to_numpy()[row_index]
    y = df[_HOLE_HAND_GRID_RAW_Y_COL].to_numpy()[row_index]
    size = HOLE_HAND_GRID_SIZE
    cols = np.rint(np.clip(x, 0.0, 1.0) * (size - 1)).astype(int)
    rows = np.rint(np.clip(y, 0.0, 1.0) * (size - 1)).astype(int)
    cell = np.where(x < 0.0, -1, rows * size + cols)
    return cell.astype(str)


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
    root_df: pd.DataFrame, filtered: pd.DataFrame, parent_node_df: pd.DataFrame | None,
    parent_group_by_keys: tuple[str, ...], graph_keys: list[str], display_keys: list[str], collapsed: bool,
    key_prefix: str = "",
) -> None:
    """One line chart per feature picked via a sub-strategy's own local
    "Add graph" control (see _render_substrategy), each paired with a
    multiselect of other features to "cross" it with, ranked strongest-
    interaction-first and labeled with that strength (see
    _decision_variance_by_key -- interaction here means "decision variance
    explained by grouping jointly with the graphed feature", the same
    metric every other ranking in this file uses) -- every feature picked
    there adds its own row of 4 heatmaps (this graphed feature as the x
    axis, the picked feature as the y axis, one heatmap per simplified
    action rate). Exact Hole Hand (_HOLE_HAND_GRID_KEY) is the one
    exception: inherently 2D already, it renders its own fixed set of
    heatmaps in place of the line chart, with no "cross" multiselect (and
    is never itself offered as something else's cross target -- it has no
    ordinary bucket-label column to pivot on, see _build_dataframe).

    `key_prefix` disambiguates one call from another when _render_substrategy
    calls this once per sub-strategy node (so every node gets its own,
    independently-computed set of graphs over just its own rows) --
    without it, two nodes' widgets for the same graphed feature would
    collide on the same Streamlit widget key."""
    if not graph_keys:
        return
    st.subheader("Graphs")

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
        interaction = _decision_variance_by_key(
            root_df, filtered, parent_node_df, parent_group_by_keys, other_keys, (key,), collapsed,
        )
        interaction_by_key = dict(interaction)
        # Strongest-interaction-first -- already restricted to other_keys.
        ranked_other_keys = [k for k, _ in interaction]

        def _cross_option_label(k: str) -> str:
            return f"{cfr_features.feature_label(k)}  ({interaction_by_key.get(k, 0.0):.0f}% variance explained)"

        col_controls, col_chart = st.columns([1, 3])
        with col_controls:
            st.markdown(f"**{cfr_features.feature_label(key)}**")
            cross_keys = _sticky_multiselect(
                "Cross with other features (adds heatmaps below)",
                ranked_other_keys, key=f"{key_prefix}graph_cross::{key}", default=[],
                format_func=_cross_option_label,
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


def _rows_digest(df: pd.DataFrame | None) -> str:
    """Cheap, stable hash of a dataframe's own row *content* (index plus
    every column's own values, via pandas' own vectorized row hasher --
    materializing the actual bytes only once, not per grouping) -- the
    actual st.cache_data key for _decision_variance_explained, since the
    dataframes themselves (potentially huge -- default_df/parent_node_df
    can be the whole reservoir) are excluded from Streamlit's own hashing
    via their leading-underscore parameter names there. Hashing row
    *positions* alone would be cheaper still, but isn't safe: two
    unrelated dataframes (different checkpoints, or even the same
    checkpoint reloaded) can easily share the same row-index range (e.g.
    both just 0..n-1) while holding completely different data, which would
    silently collide in the cache and return one sub-strategy's stale
    result for another's query -- exactly the bug an incomplete digest
    exists to avoid, not reintroduce. "" for None (no parent grouping in
    effect)."""
    if df is None:
        return ""
    return hashlib.sha1(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes()).hexdigest()


@st.cache_data(show_spinner=False)
def _residual_and_total_variance(
    _df: pd.DataFrame, _default_df: pd.DataFrame, _parent_node_df: pd.DataFrame | None,
    default_digest: str, parent_digest: str,
    parent_group_by_keys: tuple[str, ...], collapsed: bool,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """The part of _decision_variance_explained's own computation that
    doesn't depend on `resolved_split_by` at all -- (default_index,
    residual, total_variance), after subtracting whatever the parent's own
    grouping (`parent_group_by_keys`) already predicts (see
    _decision_variance_explained's own docstring for why). Split into its
    own cached function because _decision_variance_by_key calls
    _decision_variance_explained once per candidate key, every time with
    the same default_df/parent_node_df/parent_group_by_keys/collapsed but
    a *different* resolved_split_by -- without this split, every one of
    those calls was a full cache miss recomputing this identical,
    resolved_split_by-independent preamble (including its own two
    _group_labels_for_rows calls, one of the more expensive steps here)
    from scratch, once per candidate, instead of once per node visited.
    None if `default_df` is empty (nothing to measure)."""
    default_df, parent_node_df = _default_df, _parent_node_df
    if default_df.empty:
        return None
    action_cols = _action_columns(collapsed)
    default_view = _with_action_view(default_df, collapsed)
    default_index = default_df.index.to_numpy()
    default_actions = default_view[action_cols].to_numpy()

    if parent_group_by_keys and parent_node_df is not None and not parent_node_df.empty:
        parent_view = _with_action_view(parent_node_df, collapsed)
        parent_labels = _group_labels_for_rows(_df, parent_node_df.index.to_numpy(), parent_group_by_keys)
        parent_group_means = pd.DataFrame(parent_view[action_cols].to_numpy(), index=parent_labels).groupby(level=0).mean()
        own_parent_labels = _group_labels_for_rows(_df, default_index, parent_group_by_keys)
        # copy=True: under pandas's Copy-on-Write, .to_numpy() can hand back
        # a read-only view when reindex ends up a no-op (index already
        # matches) -- baseline is mutated in place just below, so it needs
        # its own writable buffer regardless.
        baseline = parent_group_means.reindex(own_parent_labels).to_numpy(copy=True)
        # A group observed among this node's own rows but never observed
        # in the parent's own claimed scope shouldn't normally happen
        # (this node's own rows are a subset of the parent's) -- fall back
        # to the parent's own grand mean rather than propagate a NaN.
        missing = np.isnan(baseline).any(axis=1)
        if missing.any():
            baseline[missing] = parent_view[action_cols].mean().to_numpy()
    else:
        baseline = np.broadcast_to(default_actions.mean(axis=0), default_actions.shape)

    residual = default_actions - baseline
    total_variance = float((residual ** 2).sum(axis=1).mean())
    return default_index, residual, total_variance


@st.cache_data(show_spinner="Computing decision variance explained...")
def _decision_variance_explained(
    _df: pd.DataFrame, _default_df: pd.DataFrame, _parent_node_df: pd.DataFrame | None,
    default_digest: str, parent_digest: str,
    parent_group_by_keys: tuple[str, ...], resolved_split_by: list[str], collapsed: bool,
) -> float | None:
    """"Decision variance explained by Split By features on claimed
    samples" (see _render_substrategy) -- the fraction of variance in the
    net's own predicted action-probability vectors, across this node's
    own claimed-and-not-further-claimed rows (`default_df`), that grouping
    those rows by `resolved_split_by` (this node's own chosen Split By
    feature(s)) explains -- a standard ANOVA/eta-squared "variance
    explained by grouping" statistic, computed directly on the model's own
    decisions. This is the one metric the whole file uses -- see
    _decision_variance_by_key, the thin wrapper around this function every
    "Add ..." dropdown ranking, the feature table, and every "Suggested
    sub-strategies" button is built on. Cached (`default_digest`/
    `parent_digest` -- see _rows_digest -- are the actual cache key
    standing in for `_default_df`/`_parent_node_df`, which are otherwise
    excluded from Streamlit's own hashing by their leading underscore, far
    too expensive to hash directly at this call volume): recomputing this
    same grouping on an unchanged sample is then instant on every rerun
    that doesn't actually touch it, and a fresh one shows a spinner instead
    of silently freezing the page. None if `resolved_split_by` is empty
    (nothing chosen yet to measure).

    Before measuring that, each row's own predicted vector first has
    whatever the *parent's* own Split By grouping (`parent_group_by_keys`)
    already predicts for it subtracted off -- computed from
    `parent_node_df` (the parent's own full claimed scope, not just this
    node's narrower `default_df`; see _resolve_node_df), i.e. the "global/
    default strategy" already implied by having reached this node at all.
    That keeps a node that happens to share its parent's own Split By
    feature honest: if the feature's relationship to the decision is the
    same within this node's own claimed rows as it is across the parent's
    whole domain, there's nothing *left* for grouping by it again to
    explain (this reads ~0%, correctly) -- but if this node's own narrower
    claim genuinely interacts with that feature (a locally different
    relationship), grouping the leftover residual by it *does* explain a
    real share of what remains (reads meaningfully above 0%, also
    correctly). No parent grouping in effect (root, or a parent with no
    Split By chosen) leaves the baseline as the plain grand mean --
    ordinary, unadjusted ANOVA.

    Each row is scored against its own `resolved_split_by` group's
    leave-one-out mean, not the plain in-sample group mean: fitting a
    group's mean on the same rows it's then scored against lets a group
    "explain" its own members almost perfectly just by being small,
    regardless of whether the grouping feature carries any real signal --
    a singleton group has an in-sample deviation of exactly 0 by
    construction. Left uncorrected, a candidate feature that's
    overwhelmingly one value with only a handful of stray outlier rows
    (colloquially "univariate," even though technically >= 2 categories
    are observed) can score *higher* than a genuinely informative
    candidate purely by carving those outliers into tiny/singleton groups
    -- see _add_best_second_split_by, which picks whichever candidate
    scores highest here. 0.0 (not None -- there's something to measure,
    it's just empty) if `default_df` has no rows, e.g. a sub-strategy
    every one of whose children together claims its entire own scope,
    leaving nothing of its own "default behaviour" left to explain."""
    if not resolved_split_by:
        return None
    shared = _residual_and_total_variance(
        _df, _default_df, _parent_node_df, default_digest, parent_digest, parent_group_by_keys, collapsed,
    )
    if shared is None:
        return 0.0
    default_index, residual, total_variance = shared
    if total_variance <= 0.0:
        return 0.0

    own_labels = _group_labels_for_rows(_df, default_index, tuple(resolved_split_by))
    own_group_means = pd.DataFrame(residual, index=own_labels).groupby(level=0).transform("mean").to_numpy()
    in_sample_deviation = residual - own_group_means

    # Leave-one-out deviation has a closed form for a group of size n_g >=
    # 2: n_g / (n_g - 1) times the in-sample deviation above (algebraically,
    # r_i - (group_sum - r_i) / (n_g - 1) == n_g / (n_g - 1) * (r_i -
    # group_mean)). A singleton group (n_g == 1) has no leave-one-out
    # estimate at all -- there's no *other* member left to average -- so it
    # gets no explanatory credit: its deviation reverts to the full,
    # pre-grouping residual (as if this row's own group hadn't explained
    # anything).
    group_sizes = pd.Series(own_labels).value_counts().reindex(own_labels).to_numpy()
    singleton = group_sizes <= 1
    # A singleton's own scale factor is never actually used below (np.where
    # picks the plain-residual branch for it instead) -- its denominator is
    # only kept away from a real 0 here so computing the (otherwise-unused)
    # scaled branch doesn't raise a division-by-zero warning.
    loo_scale = group_sizes / np.where(singleton, 1, group_sizes - 1)
    loo_deviation = np.where(singleton[:, None], residual, in_sample_deviation * loo_scale[:, None])
    within_group_variance = float((loo_deviation ** 2).sum(axis=1).mean())

    return max(0.0, 1.0 - within_group_variance / total_variance) * 100


def _decision_variance_by_key(
    root_df: pd.DataFrame, pool_df: pd.DataFrame, parent_node_df: pd.DataFrame | None,
    parent_group_by_keys: tuple[str, ...], candidate_keys: list[str], pair_with: tuple[str, ...],
    collapsed: bool,
) -> list[tuple[str, float]]:
    """One (key, pct) pair per `candidate_keys` entry, strongest-first --
    the single metric behind every "Add ..." dropdown ranking, the feature
    table, and every "Suggested sub-strategies" button in this file (see
    the module docstring): `pct` is _decision_variance_explained(root_df,
    pool_df, parent_node_df, parent_group_by_keys, [*pair_with, key],
    collapsed) -- plain "how much does this feature alone explain" when
    `pair_with` is empty, or "how much do this feature and whatever's
    already in `pair_with` explain *together*" otherwise (used to rank a
    second Split By candidate, a cross-graph target, or a child-split
    candidate against the current Split By -- see each caller). 0.0 for a
    key already in `pair_with` (nothing left for it to add).

    Exact Hole Hand (_HOLE_HAND_GRID_KEY), whether in `candidate_keys` or
    `pair_with`, is not a special case here: _group_labels_for_rows can
    jointly group it with any other key (its own 13x13 grid cell alongside
    each other key's ordinary bucket label), so `[*pair_with, key]` is
    exactly as well-defined as it is for any other feature -- even though
    a person could never adopt that specific combination as their own
    Split By pick (it always fills both of Split By's own slots by itself
    -- see _resolve_split_by), the joint decision variance it and another
    feature explain *together* is still a real, answerable question, and
    every caller that means "what would Split By become if I picked this"
    rather than "how much does this explain jointly" asks it a different
    way instead (see _add_optimise_split_by's own solo evaluation of it,
    and _splittable_candidates' `include_hole_hand_grid`, which
    _add_best_second_split_by deliberately never passes)."""
    pool_digest = _rows_digest(pool_df)
    parent_digest = _rows_digest(parent_node_df)
    scores = {}
    for key in candidate_keys:
        if key in pair_with:
            scores[key] = 0.0
            continue
        pct = _decision_variance_explained(
            root_df, pool_df, parent_node_df, pool_digest, parent_digest,
            parent_group_by_keys, [*pair_with, key], collapsed,
        )
        scores[key] = pct if pct is not None else 0.0
    return sorted(scores.items(), key=lambda kv: -kv[1])


_SUBSTRATEGY_CHILDREN_STATE_KEY = "substrategy_children"
_SELECTED_SUBSTRATEGY_STATE_KEY = "selected_substrategy"


def _substrategy_children(key_prefix: str) -> list[str]:
    """Ordered child sub-strategy ids currently nested directly under the
    node at `key_prefix`, persisted in st.session_state -- unlike the old
    automatic one-heading-per-observed-value nesting, which sub-strategies
    exist and in what priority order is now explicit, user-managed state
    (added one at a time via "Add sub-strategy"), not something derivable
    from the data alone."""
    return st.session_state.setdefault(_SUBSTRATEGY_CHILDREN_STATE_KEY, {}).setdefault(key_prefix, [])


def _child_key_prefix(parent_key_prefix: str, child_id: str) -> str:
    return f"{parent_key_prefix}substrategy_{child_id}::"


def _parent_key_prefix_and_child_id(key_prefix: str) -> tuple[str | None, str | None]:
    """(None, None) for root ("root::", which has no parent); otherwise
    (parent_key_prefix, child_id), split off `key_prefix`'s own last
    "substrategy_<id>::" segment. Purely a string operation -- key_prefix
    is always built from "root::" plus one such segment per generation
    (see _child_key_prefix/_add_substrategy), never user input."""
    if key_prefix == "root::":
        return None, None
    parent_prefix, _, tail = key_prefix.rpartition("substrategy_")
    return parent_prefix, tail.rstrip(":")


def _ancestor_prefixes(key_prefix: str) -> list[str]:
    """Every prefix from "root::" up to and including `key_prefix` itself,
    in root-to-leaf order -- e.g. "root::substrategy_ab::substrategy_cd::"
    -> ["root::", "root::substrategy_ab::", "root::substrategy_ab::substrategy_cd::"]."""
    prefixes = [key_prefix]
    while True:
        parent, _ = _parent_key_prefix_and_child_id(prefixes[-1])
        if parent is None:
            break
        prefixes.append(parent)
    return list(reversed(prefixes))


def _selected_node() -> str:
    return st.session_state.get(_SELECTED_SUBSTRATEGY_STATE_KEY, "root::")


def _select_node(key_prefix: str) -> None:
    st.session_state[_SELECTED_SUBSTRATEGY_STATE_KEY] = key_prefix


def _add_substrategy(key_prefix: str) -> None:
    children = _substrategy_children(key_prefix)
    new_id = uuid.uuid4().hex[:8]
    children.append(new_id)
    # Jump straight to the new sub-strategy -- otherwise "Add sub-strategy"
    # would leave the view unchanged (only the currently selected node ever
    # renders -- see _render_substrategy), with no visible sign anything
    # happened short of checking the sidebar.
    _select_node(_child_key_prefix(key_prefix, new_id))


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
    removed_prefix = _child_key_prefix(parent_key_prefix, child_id)
    # If the removed node (or any descendant of it) was the one currently
    # on screen, its own key_prefix no longer refers to anything -- back
    # out to the parent rather than leave the view pointed at a node that
    # no longer exists.
    if _selected_node().startswith(removed_prefix):
        _select_node(parent_key_prefix)


def _move_substrategy(parent_key_prefix: str, child_id: str, delta: int) -> None:
    children = _substrategy_children(parent_key_prefix)
    i = children.index(child_id)
    j = i + delta
    if 0 <= j < len(children):
        children[i], children[j] = children[j], children[i]


# Streamlit forgets a *widget's own* session_state entry the moment its
# st.xxx(key=...) call stops executing for even one script run (e.g. a
# sub-strategy node that isn't the one currently selected -- confirmed
# directly: st.session_state[widget_key] raises KeyError, "did you forget
# to initialize it", after a run where that widget's call site didn't
# execute). Since only the selected node's own widgets render each run
# (see _render_substrategy), that would silently reset every OTHER node's
# own claim filters/Split By pick/extras back to their defaults the moment
# you navigate away from them -- breaking both the waterfall row-claiming
# math (an ancestor's or earlier sibling's claim, no longer "remembered",
# would stop being subtracted at all) and the UI (a revisited node's
# controls would appear to have forgotten your earlier choices). The fix:
# every widget below that some OTHER node's computation or a later
# revisit depends on is "sticky" -- its current value is copied into one
# of these plain (non-widget) session_state dicts right after it renders,
# and that copy (not the widget's own, unreliable session_state entry) is
# what both this node's *own* `default=` on a later re-render, and any
# OTHER node's read of it, actually use.
_SUBSTRATEGY_CLAIMS_STATE_KEY = "substrategy_claims"
_SUBSTRATEGY_SPLIT_BY_STATE_KEY = "substrategy_split_by"
_WIDGET_VALUE_SHADOW_STATE_KEY = "widget_value_shadow"


def _local_filters_from_state(key_prefix: str) -> dict[str, list[str]]:
    """This node's own local claim filters -- the sticky copy persisted
    the last time its real "Add filter"/"keep values" widgets rendered
    (see the module-level comment above and _render_substrategy). `{}`
    (claims everything, its own true default) for a node that's never
    been visited yet."""
    return st.session_state.setdefault(_SUBSTRATEGY_CLAIMS_STATE_KEY, {}).get(key_prefix, {})


def _set_local_filters(key_prefix: str, filters: dict[str, list[str]]) -> None:
    st.session_state.setdefault(_SUBSTRATEGY_CLAIMS_STATE_KEY, {})[key_prefix] = filters


def _split_by_from_state(key_prefix: str) -> list[str]:
    """This node's own (unresolved, possibly not-yet-capped-at-2) Split By
    pick -- the sticky copy persisted the last time its real Split By
    widget rendered. `[]` for root or a node that's never been visited."""
    return st.session_state.setdefault(_SUBSTRATEGY_SPLIT_BY_STATE_KEY, {}).get(key_prefix, [])


def _set_split_by(key_prefix: str, split_by: list[str]) -> None:
    st.session_state.setdefault(_SUBSTRATEGY_SPLIT_BY_STATE_KEY, {})[key_prefix] = split_by


# Save & load: a saved strategy is exactly the tree structure the three
# sticky stores above hold -- which nodes exist and in what nesting/
# priority order, each one's own claim filters, and each one's own Split
# By pick -- round-tripped through a small JSON file per checkpoint, on
# top of an always-on autosave/autoload of the same shape so a session
# resumes where it left off with no deliberate action required (see
# _autosave_strategy/_autoload_strategy_once, both called once per run
# from main). Deliberately excludes the "purely illustrative" Add table/
# Add graph extras (_WIDGET_VALUE_SHADOW_STATE_KEY) -- those aren't part
# of a sub-strategy's own implementable rule, just scratch analysis.
_STRATEGY_SAVE_EXTENSION = ".json"
_AUTOSAVE_STRATEGY_NAME = "_autosave"
_STRATEGY_AUTOSAVE_DIGEST_STATE_KEY = "strategy_autosave_digest"
_STRATEGY_AUTOLOAD_CHECKPOINT_STATE_KEY = "strategy_autoload_checkpoint"


def _strategies_dir(checkpoint_path: str) -> str:
    """Where this checkpoint's own saved sub-strategy-tree structures (see
    _current_strategy_state) live on disk -- colocated with, and named
    after, `checkpoint_path` itself (not one shared global directory),
    since a saved tree's own claim filters and Split By picks name feature
    keys and bucket values that are only meaningful for the checkpoint
    they were built against. Overridable via CFR_EXPLORER_STRATEGIES_DIR
    (mirrors _default_checkpoint_path's own env-var override) purely so
    tests/test_cfr_explorer.py can point every test at its own isolated
    scratch directory -- several tests share one on-disk checkpoint path
    (module-scoped fixtures, keeping slow-to-build net inference results
    cached across tests -- see _synthetic_checkpoint_path), which would
    otherwise leak one test's saved/autosaved strategies into the next.
    Real usage never sets this, and always gets one scoped to its own
    checkpoint_path instead."""
    override = os.environ.get("CFR_EXPLORER_STRATEGIES_DIR")
    return override if override else f"{checkpoint_path}.explorer_strategies"


def _sanitize_strategy_name(name: str) -> str:
    """A user-typed "Save as" name, restricted to characters safe to use
    verbatim as a filename -- prevents directory traversal (a name like
    "../../etc/passwd" collapses to just "etcpasswd")."""
    return re.sub(r"[^A-Za-z0-9_ -]", "", name).strip()


def _strategy_path(checkpoint_path: str, name: str) -> str:
    return os.path.join(_strategies_dir(checkpoint_path), f"{name}{_STRATEGY_SAVE_EXTENSION}")


def _saved_strategy_names(checkpoint_path: str) -> list[str]:
    """Every named (i.e. not the reserved autosave) strategy currently
    saved for `checkpoint_path`, sorted alphabetically -- the options
    "Load a saved strategy" offers. [] if nothing's been saved yet (no
    directory to list)."""
    strategies_dir = _strategies_dir(checkpoint_path)
    if not os.path.isdir(strategies_dir):
        return []
    names = [
        filename[: -len(_STRATEGY_SAVE_EXTENSION)]
        for filename in os.listdir(strategies_dir)
        if filename.endswith(_STRATEGY_SAVE_EXTENSION)
    ]
    return sorted(name for name in names if name != _AUTOSAVE_STRATEGY_NAME)


def _node_exists(key_prefix: str, children_by_prefix: dict[str, list[str]]) -> bool:
    """Whether `key_prefix` is still reachable from root through
    `children_by_prefix` -- used to validate a saved "selected" node (see
    _current_strategy_state/_apply_strategy_state) still exists in the
    tree being restored, rather than blindly trusting a value that could
    point at a node a since-edited tree no longer has."""
    if key_prefix == "root::":
        return True
    parent_prefix, child_id = _parent_key_prefix_and_child_id(key_prefix)
    if parent_prefix is None:
        return False
    return child_id in children_by_prefix.get(parent_prefix, []) and _node_exists(parent_prefix, children_by_prefix)


def _current_strategy_state() -> dict:
    """This session's own sub-strategy tree structure -- exactly, and
    only, what "Save"/autosave persist and "Load" restores: which nodes
    exist and in what nesting/priority order (`children`), each one's own
    claim filters (`claims`) and Split By pick (`split_by`), plus whichever
    node is currently on screen (`selected`), so reopening a saved
    strategy also returns you to where you left off within it."""
    return {
        "children": st.session_state.setdefault(_SUBSTRATEGY_CHILDREN_STATE_KEY, {}),
        "claims": st.session_state.setdefault(_SUBSTRATEGY_CLAIMS_STATE_KEY, {}),
        "split_by": st.session_state.setdefault(_SUBSTRATEGY_SPLIT_BY_STATE_KEY, {}),
        "selected": _selected_node(),
    }


def _strategy_state_digest(state: dict) -> str:
    return hashlib.sha1(json.dumps(state, sort_keys=True).encode("utf-8")).hexdigest()


def _apply_strategy_state(state: dict) -> None:
    """Overwrites this session's own sub-strategy tree with `state` (as
    produced by _current_strategy_state, then round-tripped through JSON
    -- see _save_strategy/_load_strategy), and clears every currently
    -instantiated per-node widget's own session_state entry (every key
    starting with the shared "root::" prefix every node's own key_prefix
    is built from -- see _child_key_prefix) so each one re-initializes
    from the newly loaded sticky state on its own next render, rather than
    an already-instantiated widget silently keeping its own stale value (a
    widget's `default=` is only ever honored the very first time its key
    is instantiated -- see the module-level comment on the sticky-state
    stores, and _add_best_second_split_by for the same fix applied to one
    single widget instead of every one at once)."""
    children = state.get("children", {})
    st.session_state[_SUBSTRATEGY_CHILDREN_STATE_KEY] = children
    st.session_state[_SUBSTRATEGY_CLAIMS_STATE_KEY] = state.get("claims", {})
    st.session_state[_SUBSTRATEGY_SPLIT_BY_STATE_KEY] = state.get("split_by", {})
    for key in [k for k in st.session_state if k.startswith("root::")]:
        del st.session_state[key]
    selected = state.get("selected", "root::")
    _select_node(selected if _node_exists(selected, children) else "root::")


def _save_strategy(checkpoint_path: str, name: str) -> None:
    strategies_dir = _strategies_dir(checkpoint_path)
    os.makedirs(strategies_dir, exist_ok=True)
    with open(_strategy_path(checkpoint_path, name), "w", encoding="utf-8") as f:
        json.dump(_current_strategy_state(), f, indent=2)


def _load_strategy(checkpoint_path: str, name: str) -> None:
    path = _strategy_path(checkpoint_path, name)
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        state = json.load(f)
    _apply_strategy_state(state)


def _autoload_strategy_once(checkpoint_path: str) -> None:
    """Restores this checkpoint's own most recent autosave (see
    _autosave_strategy) the first time `checkpoint_path` is seen in this
    session -- "starting up" pointed at a checkpoint that already has one
    resumes exactly where that autosave left off, rather than an empty
    tree, without repeatedly re-applying it (clobbering live edits) on
    every later rerun. Guarded per checkpoint_path, not just "ever done
    once", so switching to a different checkpoint via the sidebar's own
    text input also resumes *that* checkpoint's own most recent autosave
    the first time it's selected.

    Clears the autosave-change-detection baseline (see _autosave_strategy)
    rather than establishing it here directly -- every node's own widgets
    still stamp a "no filters"/"no Split By yet" entry into the sticky
    stores the very first time they're rendered, even when nothing was
    actually picked (see _set_local_filters/_set_split_by, called
    unconditionally), so the state immediately after this first render
    would otherwise almost always read as "different" from whatever was
    just loaded (or from the untouched empty default) purely from that
    bookkeeping -- not a real change -- and get needlessly (if
    harmlessly) written straight back out before anyone's done anything.
    Leaving the baseline unset instead means _autosave_strategy's own
    first call this checkpoint just quietly establishes it against
    whatever this first render actually produces."""
    if st.session_state.get(_STRATEGY_AUTOLOAD_CHECKPOINT_STATE_KEY) == checkpoint_path:
        return
    st.session_state[_STRATEGY_AUTOLOAD_CHECKPOINT_STATE_KEY] = checkpoint_path
    _load_strategy(checkpoint_path, _AUTOSAVE_STRATEGY_NAME)
    st.session_state.pop(_STRATEGY_AUTOSAVE_DIGEST_STATE_KEY, None)


def _autosave_strategy(checkpoint_path: str) -> None:
    """Silently persists this session's own current sub-strategy tree (see
    _current_strategy_state) as this checkpoint's own autosave, but only
    when it's actually different from the baseline established the first
    time this checkpoint was seen this session (tracked via a plain
    content digest, not a dirty flag threaded through every mutating
    callback -- cheaper to compute once per render than to keep correct at
    every single call site that can change the tree; see
    _autoload_strategy_once for why that baseline is established here, on
    this function's own first call, rather than any earlier). Called once
    per script run, after every other widget/callback has already had its
    chance to mutate state this run (see main), so it always reflects this
    run's own final state."""
    state = _current_strategy_state()
    digest = _strategy_state_digest(state)
    previous_digest = st.session_state.get(_STRATEGY_AUTOSAVE_DIGEST_STATE_KEY)
    st.session_state[_STRATEGY_AUTOSAVE_DIGEST_STATE_KEY] = digest
    if previous_digest is None or previous_digest == digest:
        return
    strategies_dir = _strategies_dir(checkpoint_path)
    os.makedirs(strategies_dir, exist_ok=True)
    with open(_strategy_path(checkpoint_path, _AUTOSAVE_STRATEGY_NAME), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _render_strategy_controls(checkpoint_path: str) -> None:
    """Sidebar "Strategy" section: explicit named Save/Load for this
    checkpoint's own saved sub-strategy-tree structures, on top of (not
    instead of) the automatic autosave/autoload every session already gets
    regardless (see _autosave_strategy/_autoload_strategy_once, both
    called from main) -- named saves are for deliberately keeping several
    distinct trees around (e.g. one implementable strategy per stack
    depth), while autosave/autoload alone is what makes closing and
    reopening the app resume the very last thing worked on with no
    deliberate action required."""
    st.sidebar.subheader("Strategy")
    st.sidebar.caption(
        "Every change here autosaves for this checkpoint, and resumes automatically next time -- "
        "use Save/Load below only to keep multiple named strategies around."
    )
    saved_names = _saved_strategy_names(checkpoint_path)
    load_choice = st.sidebar.selectbox(
        "Load a saved strategy", options=["Choose a saved strategy..."] + saved_names, key="strategy_load_choice",
    )
    st.sidebar.button(
        "Load", key="strategy_load_button", disabled=load_choice == "Choose a saved strategy...",
        on_click=_load_strategy, args=(checkpoint_path, load_choice),
    )
    save_name = st.sidebar.text_input("Save current strategy as", key="strategy_save_name")
    sanitized_save_name = _sanitize_strategy_name(save_name)
    st.sidebar.button(
        "Save", key="strategy_save_button", disabled=not sanitized_save_name,
        on_click=_save_strategy, args=(checkpoint_path, sanitized_save_name),
    )


def _sticky_multiselect(label: str, options: list[str], key: str, default: list[str], **kwargs) -> list[str]:
    """st.multiselect that remembers its own value across the widget not
    rendering for a run or more, via _WIDGET_VALUE_SHADOW_STATE_KEY (see
    the module-level comment above) -- for widgets whose value nothing
    *else* needs to read (unlike claim filters/Split By, which get their
    own dedicated sticky stores above so their reads can skip the
    `default=` fallback dance this does). `default` is this widget's
    fallback the *first* time it's ever instantiated (no sticky value yet)
    -- e.g. "every observed value" for a fresh "keep values" widget."""
    shadow = st.session_state.setdefault(_WIDGET_VALUE_SHADOW_STATE_KEY, {})
    previous = [v for v in shadow.get(key, default) if v in options]
    value = st.multiselect(label, options=options, default=previous, key=key, **kwargs)
    shadow[key] = value
    return value


def _resolve_incoming_df(root_df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    """The dataframe pool visible to the node at `key_prefix`, BEFORE its
    own local claim filters -- `root_df` itself for root, or otherwise its
    parent's own claimed scope minus whatever earlier siblings (at every
    ancestor level along the way) have already claimed. Purely a
    st.session_state walk (see _local_filters_from_state), no widgets
    rendered -- lets the currently selected node's own real widgets start
    from the right pool without any of its ancestors needing to render too
    (only the selected node renders -- see _render_substrategy)."""
    prefixes = _ancestor_prefixes(key_prefix)  # root-to-leaf, key_prefix itself last
    node_df = root_df  # becomes "the previous ancestor's own node_df" each iteration
    incoming_df = root_df
    for prefix in prefixes:
        parent_prefix, this_id = _parent_key_prefix_and_child_id(prefix)
        if parent_prefix is None:
            incoming_df = root_df
        else:
            remaining = node_df
            for sibling_id in _substrategy_children(parent_prefix):
                if sibling_id == this_id:
                    break
                sibling_prefix = _child_key_prefix(parent_prefix, sibling_id)
                claimed = _apply_filters(remaining, _local_filters_from_state(sibling_prefix))
                remaining = remaining.drop(claimed.index)
            incoming_df = remaining
        if prefix == key_prefix:
            return incoming_df
        node_df = _apply_filters(incoming_df, _local_filters_from_state(prefix))
    return incoming_df


def _resolve_node_df(root_df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    """A node's own full claimed scope -- _resolve_incoming_df's pool, with
    that same node's own local claim filters applied on top (the one
    further step _resolve_incoming_df deliberately stops short of, since
    the currently selected node applies its own filters via its real,
    interactive widgets instead). Used to find a node's *parent's* own
    full claimed scope: the `parent_node_df` a sub-strategy's own
    _decision_variance_explained/_decision_variance_by_key calls draw their
    baseline from when correcting for a grouping the parent's Split By
    already applies (see _render_substrategy) -- deliberately the parent's
    full node_df, not its narrower default_df
    (which excludes every child, including whichever one is asking), since
    the parent's own Split By is conceptually a rule over its whole
    domain, not just whatever's left after every child has carved its own
    share out of it."""
    incoming_df = _resolve_incoming_df(root_df, key_prefix)
    return _apply_filters(incoming_df, _local_filters_from_state(key_prefix))


def _resolve_default_df(node_df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    """`node_df` (the currently selected node's own full claimed scope)
    minus whatever its own *direct* children have already claimed (via
    _local_filters_from_state, one per child) -- this node's actual
    "default behaviour" leftover, without rendering any child (only the
    selected node renders -- see _render_substrategy)."""
    remaining = node_df
    for child_id in _substrategy_children(key_prefix):
        child_prefix = _child_key_prefix(key_prefix, child_id)
        claimed = _apply_filters(remaining, _local_filters_from_state(child_prefix))
        remaining = remaining.drop(claimed.index)
    return remaining


def _capped_for_eval(df: pd.DataFrame, max_eval_samples: int) -> pd.DataFrame:
    """`df` itself if it's already at or under `max_eval_samples` rows,
    otherwise a fixed-seed random subsample of exactly `max_eval_samples`
    of them -- the actual sample every table/chart/metric/ranking/button
    in this file computes over (see _render_substrategy), decoupled from
    however many rows a node's own claim genuinely matches (`node_df`/
    `default_df` themselves, used unchanged for heading counts and "keep
    values" filter options -- see _render_substrategy). A *fixed* seed
    (not fresh per render) so a spot's own analysis reads the same numbers
    across reruns that don't actually change its own claimed rows, rather
    than jittering on every unrelated interaction elsewhere on the page."""
    if len(df) <= max_eval_samples:
        return df
    return df.sample(n=max_eval_samples, random_state=0)


def _add_children_for_each_level(key_prefix: str, feature_key: str, default_df: pd.DataFrame, split_by: list[str]) -> None:
    """One new child sub-strategy per level of `feature_key` actually
    observed in `default_df` -- each claiming exactly that level, with
    Split By set to `split_by` (typically the parent's own current pick,
    so every new child keeps examining the same breakdown within its own
    narrower slice). Shared by _add_max_interaction_split/
    _add_max_importance_split below -- they differ only in *which*
    feature they pick, not in what happens once one's chosen."""
    children = _substrategy_children(key_prefix)
    for level in _observed_categories(default_df, feature_key):
        child_id = uuid.uuid4().hex[:8]
        children.append(child_id)
        child_prefix = _child_key_prefix(key_prefix, child_id)
        _set_local_filters(child_prefix, {feature_key: [level]})
        _set_split_by(child_prefix, list(split_by))


def _splittable_candidates(
    default_df: pd.DataFrame, split_by_options: list[str], exclude: list[str],
    include_hole_hand_grid: bool = False,
) -> list[str]:
    """`split_by_options` minus `exclude` and Exact Hole Hand (no ordinary
    per-level claim makes sense for its own 2D grid -- see
    _HOLE_HAND_GRID_KEY), minus any feature that's constant across
    `default_df` (fewer than 2 observed levels -- splitting on it would
    just recreate a single child claiming everything, not an actual
    split) -- the pool _add_max_interaction_split/_add_max_importance_split
    pick their one candidate from.

    `include_hole_hand_grid` opts Exact Hole Hand back in -- appended, not
    subject to the same >=2-observed-levels check (_observed_categories
    can't compute one for it anyway; see _hole_hand_grid_split_by_available
    instead) -- for _add_optimise_split_by alone, the one caller that
    searches for the single best pick from scratch rather than adding to
    or claiming levels of one already in place: picking Exact Hole Hand
    there just replaces whatever Split By already held (see
    _resolve_split_by), unlike _add_best_second_split_by (which is adding
    a *second* feature alongside one already chosen -- no room left for
    something that needs both slots to itself, so it never passes this)
    or the two _add_children_for_each_level-based buttons (no way to
    filter a child down to one exact grid cell)."""
    candidates = [
        k for k in split_by_options
        if k not in exclude and k != _HOLE_HAND_GRID_KEY and len(_observed_categories(default_df, k)) >= 2
    ]
    if (
        include_hole_hand_grid and _HOLE_HAND_GRID_KEY in split_by_options
        and _HOLE_HAND_GRID_KEY not in exclude and _hole_hand_grid_split_by_available(default_df)
    ):
        candidates.append(_HOLE_HAND_GRID_KEY)
    return candidates


def _add_max_interaction_split(
    key_prefix: str, root_df: pd.DataFrame, default_df: pd.DataFrame, parent_node_df: pd.DataFrame | None,
    parent_group_by_keys: tuple[str, ...], collapsed: bool, resolved_split_by: list[str],
    split_by_options: list[str],
) -> None:
    """"Add maximum interaction split": finds the single candidate feature
    (see _splittable_candidates) with the highest *marginal* decision
    variance explained once grouped jointly with this node's own current
    Split By pick, on top of what that Split By pick already explains
    alone (see _decision_variance_by_key/_decision_variance_explained) --
    that marginal gain divided by how many children claiming it would
    actually add (see _observed_categories), like _add_max_importance_split's
    own per-level normalization -- then adds one child per level it takes
    among this node's own default rows (see _add_children_for_each_level).

    Normalizing the *marginal* gain, not the joint total, matters here
    specifically because the joint total always also includes whatever the
    current Split By pick already explains *by itself* -- a constant
    across every candidate, but one that dividing by a candidate's own
    (varying) level count doesn't treat as constant: dividing a large
    already-explained-anyway baseline by a small level count can
    trivially outscore a real, informative candidate's own genuinely
    smaller marginal contribution divided by a larger one, favoring a
    candidate that adds nothing beyond what was already known just because
    it happens to have few levels. Subtracting that shared baseline first
    removes it from the comparison entirely, leaving only what each
    candidate actually, distinctly contributes -- exactly the numerator
    _add_max_importance_split already effectively uses (see its own
    docstring): with no existing Split By to net out there, its own solo
    variance explained *is* already the marginal gain over nothing.

    A quick way to check "is there a feature whose relationship to my
    current Split By isn't just additive -- does my chosen breakdown
    actually behave differently depending on its own value" without
    manually trying each remaining feature as a filter one at a time --
    and, thanks to the per-level normalization, biased toward whichever
    such feature asks a person to learn the fewest additional
    sub-strategies for that interaction. No-op if this node has no Split
    By chosen yet (nothing to measure interaction against) or no eligible
    candidate remains."""
    if not resolved_split_by:
        return
    candidates = _splittable_candidates(default_df, split_by_options, resolved_split_by)
    if not candidates:
        return
    scores = dict(_decision_variance_by_key(
        root_df, default_df, parent_node_df, parent_group_by_keys, candidates, tuple(resolved_split_by), collapsed,
    ))
    baseline_pct = _decision_variance_explained(
        root_df, default_df, parent_node_df, _rows_digest(default_df), _rows_digest(parent_node_df),
        parent_group_by_keys, list(resolved_split_by), collapsed,
    ) or 0.0
    best_key = max(
        candidates,
        key=lambda k: max(0.0, scores[k] - baseline_pct) / len(_observed_categories(default_df, k)),
    )
    _add_children_for_each_level(key_prefix, best_key, default_df, resolved_split_by)


def _add_max_importance_split(
    key_prefix: str, default_df: pd.DataFrame, resolved_split_by: list[str],
    default_importance: list[tuple[str, float]],
) -> None:
    """"Add maximum importance split": adds one child per level of
    whichever candidate feature (see _splittable_candidates) has the
    highest decision variance explained on its own (see
    _decision_variance_by_key) *per observed level* -- raw importance
    divided by how many children claiming it would actually add (see
    _observed_categories) -- rather than raw importance alone. A feature
    that's a little more important overall but spreads that importance
    across many levels asks a person to learn and remember many more
    sub-strategies for roughly the same payoff per level; this
    approximates "biggest remaining lever per additional sub-strategy a
    person has to learn," not just "biggest lever, full stop" (unlike
    _add_max_interaction_split, which specifically looks for features that
    interact with the current Split By rather than simply mattering a lot
    on their own). `default_importance` is always each candidate's own
    solo variance explained *ignoring* whatever Split By is already
    chosen (see its own `pair_with=()` in _render_substrategy), so unlike
    _add_max_interaction_split's own joint total, there's no shared
    already-explained-by-something-else baseline riding along in the
    numerator here to strip out first -- a candidate's raw importance
    *is* already its own marginal contribution over nothing, so dividing
    it directly by level count is correct as written. No-op if no
    eligible candidate remains."""
    ranked_keys = [k for k, _ in default_importance]
    importance_by_key = dict(default_importance)
    candidates = _splittable_candidates(default_df, ranked_keys, resolved_split_by)
    if not candidates:
        return
    best_key = max(
        candidates,
        key=lambda k: importance_by_key[k] / len(_observed_categories(default_df, k)),
    )
    _add_children_for_each_level(key_prefix, best_key, default_df, resolved_split_by)


def _add_best_second_split_by(
    key_prefix: str, root_df: pd.DataFrame, default_df: pd.DataFrame, parent_node_df: pd.DataFrame | None,
    parent_group_by_keys: tuple[str, ...], collapsed: bool, resolved_split_by: list[str], split_by_options: list[str],
) -> None:
    """"Add best second Split By feature": when this node currently has
    exactly one Split By feature chosen, adds whichever second feature (out
    of every other _splittable_candidates candidate -- so, as with the
    other two buttons here, never one that's constant across this node's
    own claimed default_df, respecting whatever local filters got it there)
    maximizes "Decision variance explained by Split By features on claimed
    samples" once paired with it (see _decision_variance_by_key) --
    optimizes that exact metric directly, with no per-level normalization
    (unlike _add_max_interaction_split/_add_max_importance_split): this
    button doesn't add any new sub-strategies for a person to learn, just a
    second dimension to the *same* node's own existing rule, so there's no
    per-level cost to weigh against. No-op unless exactly one Split By
    feature is currently chosen (nothing to pair up with 0 -- and
    MAX_SPLIT_BY_FEATURES already caps a pair at 2, so there's no "third").

    Exact Hole Hand is deliberately excluded from `candidates` here (unlike
    _add_optimise_split_by, which does include it) -- it always fills
    *both* Split By slots by itself (see _resolve_split_by), so with one
    slot already taken by whatever's in `resolved_split_by`, there's no
    room left to add it: picking it wouldn't extend this node's existing
    rule the way this button promises, it would silently discard the
    feature already chosen instead. _add_optimise_split_by has no such
    conflict, since it searches for the single best pick from scratch
    rather than adding to one already in place.

    Sets `st.session_state[f"{key_prefix}split_by"]` directly (not just
    the sticky-shadow store _set_split_by normally goes through) since
    this node's own Split By widget already exists (unlike a brand-new
    child's) -- an existing widget's `default=` is only honored the very
    first time its key is ever instantiated, so only overwriting its own
    session_state entry directly actually takes effect on the very next
    render (see the module-level comment on the sticky-state stores)."""
    if len(resolved_split_by) != 1:
        return
    first = resolved_split_by[0]
    candidates = _splittable_candidates(default_df, split_by_options, resolved_split_by)
    if not candidates:
        return
    scores = dict(_decision_variance_by_key(
        root_df, default_df, parent_node_df, parent_group_by_keys, candidates, tuple(resolved_split_by), collapsed,
    ))
    best_key = max(candidates, key=lambda k: scores[k])
    st.session_state[f"{key_prefix}split_by"] = [first, best_key]
    _set_split_by(key_prefix, [first, best_key])


def _add_optimise_split_by(
    key_prefix: str, root_df: pd.DataFrame, default_df: pd.DataFrame, parent_node_df: pd.DataFrame | None,
    parent_group_by_keys: tuple[str, ...], collapsed: bool, split_by_options: list[str],
) -> None:
    """"Optimise split by features": like _add_best_second_split_by, but
    ignores whatever Split By is currently chosen and searches every
    unordered pair of _splittable_candidates from scratch (evaluated
    against no existing pairing -- an empty `resolved_split_by`, so a
    feature constant across this node's own claimed default_df is still
    excluded, but nothing else is), setting Split By to whichever pair
    maximizes "Decision variance explained by Split By features on claimed
    samples" (see _decision_variance_explained) directly -- the single best
    *pair* this node's own claimed rows support, not just the best partner
    for whatever's already chosen. No-op if fewer than 2 eligible
    candidates remain (nothing to pair).

    Exact Hole Hand is included as a candidate (when 100% preflop -- see
    _hole_hand_grid_split_by_available), but only ever evaluated *alone*,
    never jointly grouped with another feature: not because
    _group_labels_for_rows can't do that (it can -- see
    _decision_variance_by_key), but because picking it here would actually
    *replace* whatever else got paired with it, not join it (it always
    fills both of Split By's own slots by itself -- see
    _resolve_split_by), so a joint score would misrepresent what this
    button's own pick would actually resolve to. It competes against every
    genuine 2-feature pair as a single-feature alternative instead."""
    candidates = _splittable_candidates(
        default_df, split_by_options, (), include_hole_hand_grid=True,
    )
    ordinary_candidates = [k for k in candidates if k != _HOLE_HAND_GRID_KEY]
    if len(ordinary_candidates) < 2 and _HOLE_HAND_GRID_KEY not in candidates:
        return
    default_digest = _rows_digest(default_df)
    parent_digest = _rows_digest(parent_node_df)
    best_pair, best_pct = None, -1.0

    def _try(grouping: list[str]) -> None:
        nonlocal best_pair, best_pct
        pct = _decision_variance_explained(
            root_df, default_df, parent_node_df, default_digest, parent_digest,
            parent_group_by_keys, grouping, collapsed,
        )
        if pct is not None and pct > best_pct:
            best_pair, best_pct = grouping, pct

    if _HOLE_HAND_GRID_KEY in candidates:
        _try([_HOLE_HAND_GRID_KEY])
    for i, a in enumerate(ordinary_candidates):
        for b in ordinary_candidates[i + 1:]:
            _try([a, b])
    if best_pair is not None:
        st.session_state[f"{key_prefix}split_by"] = best_pair
        _set_split_by(key_prefix, best_pair)


def _render_substrategy(
    root_df: pd.DataFrame, key_prefix: str, display_keys: list[str], collapsed: bool, max_eval_samples: int,
) -> None:
    """Renders the *currently selected* sub-strategy node's own page --
    root ("root::", heading "Overall Strategy") or any node added via "Add
    sub-strategy" (its own local filters *are* its claim condition -- see
    the module docstring) -- and nothing else: every other node in the
    tree is reached by switching selection (a sidebar nav button, one of
    this node's own "Sub-strategies" quick-jump buttons, or "Back to
    parent"), not by scrolling. `root_df` is the whole loaded reservoir,
    unfiltered -- this node's own ancestors and earlier siblings are
    resolved from st.session_state alone (see _resolve_incoming_df),
    without any of them needing to render.

    Render order: this node's own local filters (define what it claims, so
    `node_df` here is its *full* claimed scope) -> heading (+ back/
    remove/move controls for a non-root node) -> "Add sub-strategy" + a
    quick-jump button per existing child -> this node's actual "default
    behaviour" (`node_df` minus every child's own claim, via
    _resolve_default_df) -- Split By widget (own local widget, defaulting
    to inherit whatever its parent's *current* Split By pair is via a
    direct st.session_state read; root has no parent, so it just defaults
    to nothing), a feature-importance-and-interaction table, and four
    "Suggested sub-strategies" buttons that build on it algorithmically
    instead of by hand (_add_max_interaction_split/_add_max_importance_split
    add one child per level of a well-chosen feature; _add_best_second_split_by/
    _add_optimise_split_by instead extend or replace *this* node's own
    Split By pick -- see each one's own docstring), the prominent
    table+graph, and the "Decision variance explained by Split By
    features on claimed samples" metric, all computed over that leftover,
    since the prominent block *is* the default/else branch of the rule
    list -- then purely illustrative "Add table"/"Add graph" additions,
    also over that same leftover.

    Both decision-variance-explained views this node computes (`importance`,
    over its own incoming pool, for "Add filter"; `default_importance`,
    over its own claimed-and-not-further-claimed leftover, for Split
    By/the variance metric) correct for whatever grouping this node's own
    *parent* already applies (`parent_group_by_keys`) by centering against
    `parent_node_df` -- the parent's own full claimed scope (see
    _resolve_node_df), not this node's own rows (see
    _decision_variance_explained's own docstring for why). That
    distinction matters: if the correction used this node's own (possibly
    much narrower) rows instead, a feature merely *constant* within that
    narrow slice would always score 0 regardless of whether the parent's
    grouping actually explains it there -- e.g. a child sharing its
    parent's own Split By feature, filtered down to a small subset, would
    always show 0% "explained" even though that narrower rule can
    genuinely capture real, additional signal the parent's broader-
    population analysis doesn't. Centering against the parent's own full
    sample instead means a feature only reads as "already explained" when
    the parent's grouping is *uniformly* true across the parent's whole
    domain, not just within whatever this node happens to have claimed."""
    parent_key_prefix, child_id = _parent_key_prefix_and_child_id(key_prefix)
    is_root = parent_key_prefix is None
    incoming_df = _resolve_incoming_df(root_df, key_prefix)

    parent_split_by = [] if is_root else _split_by_from_state(parent_key_prefix)
    parent_group_by_keys = tuple(_resolve_split_by(parent_split_by))
    parent_node_df = None if is_root else _resolve_node_df(root_df, parent_key_prefix)
    if parent_node_df is not None:
        parent_node_df = _capped_for_eval(parent_node_df, max_eval_samples)

    importance = _decision_variance_by_key(
        root_df, _capped_for_eval(incoming_df, max_eval_samples), parent_node_df, parent_group_by_keys,
        display_keys, (), collapsed,
    )
    non_graph_options = [k for k, _ in importance if k != _HOLE_HAND_GRID_KEY]
    importance_by_key = dict(importance)

    def _claim_option_label(key: str) -> str:
        return f"{cfr_features.feature_label(key)}  ({importance_by_key[key]:.0f}% variance explained)"

    if not is_root:
        st.button("← Back to parent", key=f"{key_prefix}back", on_click=_select_node, args=(parent_key_prefix,))
        st.divider()
    previous_claim = _local_filters_from_state(key_prefix)
    filter_label = "Add filter" if is_root else "Claim condition (Add filter)"
    local_filter_keys = st.multiselect(
        filter_label, options=non_graph_options, format_func=_claim_option_label,
        default=[k for k in previous_claim if k in non_graph_options],
        key=f"{key_prefix}claim_filter_keys",
    )
    local_filters: dict[str, list[str]] = {}
    for filter_key in local_filter_keys:
        observed = _observed_categories(incoming_df, filter_key)
        previous_values = previous_claim.get(filter_key, observed)
        local_filters[filter_key] = st.multiselect(
            f"{cfr_features.feature_label(filter_key)} -- keep values", options=observed,
            default=[v for v in previous_values if v in observed],
            key=f"{key_prefix}claim_filter_values::{filter_key}",
        )
    node_df = _apply_filters(incoming_df, local_filters)
    _set_local_filters(key_prefix, local_filters)
    # Computed here (rather than just before its own default_importance use
    # below) so the heading can show it alongside node_df's own count --
    # node_df is this node's own full claimed scope (its rule's total
    # domain, unaffected by whatever its children go on to carve out of
    # it), while default_df_full is what's actually left over for its own
    # "default behaviour" once every child's claim is subtracted (see
    # _resolve_default_df) -- two genuinely different numbers a person
    # could otherwise easily conflate (e.g. a parent whose own rule covers
    # everything, node_df, while most of that has already been claimed by
    # its own children, default_df_full much smaller). default_df itself
    # -- what every table/chart/metric/ranking/button below this point
    # actually computes over -- is that same leftover further capped to
    # max_eval_samples (see _capped_for_eval), decoupled from node_df/
    # default_df_full's own true counts so a common, high-volume spot's
    # own analysis stays fast regardless of how large the loaded reservoir
    # pool is.
    default_df_full = _resolve_default_df(node_df, key_prefix)
    default_df = _capped_for_eval(default_df_full, max_eval_samples)

    claim_desc = ", ".join(
        f"{cfr_features.feature_label(k)} = {', '.join(v)}" for k, v in local_filters.items()
    )
    count_parts = [f"n={len(node_df):,}"]
    if len(default_df_full) != len(node_df):
        count_parts.append(f"{len(default_df_full):,} default")
    if len(default_df_full) > max_eval_samples:
        count_parts.append(f"evaluated on {max_eval_samples:,}")
    count_suffix = f"({', '.join(count_parts)})"
    if is_root:
        heading = f"Overall Strategy ({claim_desc})" if claim_desc else "Overall Strategy"
        st.header(f"{heading}  {count_suffix}")
    else:
        heading = claim_desc or "Sub-strategy (no filter set yet -- claims everything its parent hasn't)"
        col_heading, col_up, col_down, col_remove = st.columns([10, 1, 1, 1])
        with col_heading:
            st.header(f"{heading}  {count_suffix}")
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
        return

    st.button(
        "Add sub-strategy", key=f"{key_prefix}add_substrategy",
        on_click=_add_substrategy, args=(key_prefix,),
    )

    existing_children = _substrategy_children(key_prefix)
    if existing_children:
        st.caption("Sub-strategies (click to view):")
        child_cols = st.columns(min(len(existing_children), 4))
        for i, this_child_id in enumerate(existing_children):
            child_prefix = _child_key_prefix(key_prefix, this_child_id)
            with child_cols[i % len(child_cols)]:
                st.button(
                    _nav_label(child_prefix, False), key=f"{key_prefix}jump::{this_child_id}",
                    on_click=_select_node, args=(child_prefix,), use_container_width=True,
                )

    default_importance = _decision_variance_by_key(
        root_df, default_df, parent_node_df, parent_group_by_keys, display_keys, (), collapsed,
    )
    importance_by_key = dict(default_importance)

    def _default_option_label(key: str) -> str:
        return f"{cfr_features.feature_label(key)}  ({importance_by_key[key]:.0f}% variance explained)"

    split_by_options = [k for k, _ in default_importance]
    # The very first time this node is ever visited, default to inherit
    # whatever its parent's *current* Split By pair is (`parent_split_by`,
    # read above); once it's been visited at least once, remember its own
    # choice instead (own_previous_split_by), even if that choice happens
    # to equal an earlier default -- root has no parent to inherit from.
    own_previous_split_by = _split_by_from_state(key_prefix)
    split_by_default = own_previous_split_by if own_previous_split_by else parent_split_by
    chosen_split_by = st.multiselect(
        "Split By (this sub-strategy's own 1-2 implementable features)",
        options=split_by_options, default=[k for k in split_by_default if k in split_by_options],
        format_func=_default_option_label, max_selections=MAX_SPLIT_BY_FEATURES,
        key=f"{key_prefix}split_by",
    )
    _set_split_by(key_prefix, chosen_split_by)
    resolved_split_by = _resolve_split_by(chosen_split_by)
    if resolved_split_by != chosen_split_by:
        dropped = [k for k in chosen_split_by if k not in resolved_split_by]
        st.caption(
            f"Exact Hole Hand is inherently 2D and fills both Split By slots by itself -- "
            f"ignoring {', '.join(cfr_features.feature_label(k) for k in dropped)}."
        )

    # Every number below is _decision_variance_explained/_decision_variance_by_key
    # -- the one metric this whole page uses -- always computed over this
    # node's own default_df (or incoming_df for "Add filter", above), so
    # this table always reflects exactly the sample currently matching this
    # sub-strategy's own filters, never a broader or narrower one.
    # default_importance is already sorted strongest-first (see
    # _decision_variance_by_key) -- filtering it in place, rather than
    # dropping Exact Hole Hand and re-appending it separately, keeps it in
    # its own correctly-ranked spot instead of always trailing last
    # regardless of its actual importance.
    show_importance_table = st.checkbox(
        "Show feature importance and interaction table (computationally expensive)",
        value=False, key=f"{key_prefix}show_importance_table",
    )
    if show_importance_table:
        table_keys = [
            k for k, _ in default_importance
            if k != _HOLE_HAND_GRID_KEY or _hole_hand_grid_split_by_available(default_df)
        ]
        show_interaction = bool(resolved_split_by)
        table_interaction_by_key = dict(_decision_variance_by_key(
            root_df, default_df, parent_node_df, parent_group_by_keys, table_keys, tuple(resolved_split_by), collapsed,
        )) if show_interaction else {}
        st.caption("Feature importance and interaction with the current Split By pick:")
        st.dataframe(
            pd.DataFrame([
                {
                    "Feature": cfr_features.feature_label(key),
                    "Importance": f"{importance_by_key[key]:.0f}%",
                    "Interaction with current Split By": (
                        "—" if not show_interaction or key in resolved_split_by
                        else f"{table_interaction_by_key[key]:.0f}%"
                    ),
                }
                for key in table_keys
            ]),
            hide_index=True, use_container_width=True,
        )

    st.caption("Suggested sub-strategies:")
    suggest_cols = st.columns(4)
    with suggest_cols[0]:
        st.button(
            "Add maximum interaction split", key=f"{key_prefix}add_max_interaction_split",
            disabled=not resolved_split_by,
            help="Splits on whichever remaining feature has the highest total decision variance explained "
            "jointly with this node's own current Split By pick, per observed level -- one child per level "
            "it takes.",
            on_click=_add_max_interaction_split,
            args=(
                key_prefix, root_df, default_df, parent_node_df, parent_group_by_keys, collapsed,
                resolved_split_by, split_by_options,
            ),
        )
    with suggest_cols[1]:
        st.button(
            "Add maximum importance split", key=f"{key_prefix}add_max_importance_split",
            help="Splits on the remaining feature with the highest decision variance explained per observed "
            "level -- one child per level it takes.",
            on_click=_add_max_importance_split,
            args=(key_prefix, default_df, resolved_split_by, default_importance),
        )
    with suggest_cols[2]:
        st.button(
            "Add best second Split By feature", key=f"{key_prefix}add_best_second_split_by",
            disabled=len(resolved_split_by) != 1,
            help="Adds whichever second feature, paired with this node's current lone Split By pick, maximizes "
            "Decision variance explained.",
            on_click=_add_best_second_split_by,
            args=(
                key_prefix, root_df, default_df, parent_node_df, parent_group_by_keys, collapsed,
                resolved_split_by, split_by_options,
            ),
        )
    with suggest_cols[3]:
        st.button(
            "Optimise split by features", key=f"{key_prefix}optimise_split_by",
            help="Ignores whatever Split By is currently chosen and sets it to whichever pair of features "
            "maximizes Decision variance explained on this sub-strategy's own claimed samples.",
            on_click=_add_optimise_split_by,
            args=(key_prefix, root_df, default_df, parent_node_df, parent_group_by_keys, collapsed, split_by_options),
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
            pct = _decision_variance_explained(
                root_df, default_df, parent_node_df, _rows_digest(default_df), _rows_digest(parent_node_df),
                parent_group_by_keys, resolved_split_by, collapsed,
            )
            st.metric("Decision variance explained by Split By features on claimed samples", f"{pct:.0f}%")
        else:
            st.caption("Pick 1-2 Split By features above to define this sub-strategy's implementable rule.")

    st.caption("Additional (illustrative) analysis")
    illustrative_options = [k for k in split_by_options if k != _HOLE_HAND_GRID_KEY]
    graph_options = split_by_options if _hole_hand_grid_available(default_df) else illustrative_options
    col_graph, col_table = st.columns(2)
    with col_graph:
        extra_graph_keys = _sticky_multiselect(
            "Add graph", graph_options, key=f"{key_prefix}extra_graph", default=[],
            format_func=_default_option_label,
        )
    with col_table:
        extra_table_keys = _sticky_multiselect(
            "Add table", illustrative_options, key=f"{key_prefix}extra_table", default=[],
            format_func=_default_option_label,
        )
    for table_key in extra_table_keys:
        st.caption(f"Extra table: {cfr_features.feature_label(table_key)}")
        _render_table(default_df, [table_key], collapsed)
    _render_graphs(
        root_df, default_df, parent_node_df, parent_group_by_keys, extra_graph_keys, display_keys, collapsed,
        f"{key_prefix}extra::",
    )


def _shorthand_description(key_prefix: str) -> str:
    """The same claim-condition text this node's own heading would show if
    it were the currently selected/rendered node (see _render_substrategy's
    `claim_desc`), rebuilt from the sticky claims store (_local_filters_from_state)
    instead of that node's own (possibly long-forgotten -- see the
    module-level comment above _local_filters_from_state) widget state --
    used for every OTHER node's own label in the sidebar nav tree and the
    "Sub-strategies" quick-jump buttons, none of which actually render
    that node's own widgets this script pass."""
    parts = [
        f"{cfr_features.feature_label(key)} = {', '.join(values)}"
        for key, values in _local_filters_from_state(key_prefix).items()
    ]
    return ", ".join(parts)


def _truncate(text: str, limit: int = 40) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _nav_label(key_prefix: str, is_root: bool) -> str:
    desc = _truncate(_shorthand_description(key_prefix))
    if is_root:
        return f"Overall Strategy ({desc})" if desc else "Overall Strategy"
    return desc or "Sub-strategy (unclaimed)"


# Per-depth indentation for the sidebar nav tree (see _render_navigation_node).
# A previous version drew an ASCII tree (glyphs like "│├└─") as plain text in
# front of each button's own label, the same convention any CLI `tree`
# command uses -- but Streamlit centers a button's label within it, so a
# longer label re-centers the whole string and drags its glyphs out of
# alignment with everything else, defeating the entire point. Real CSS
# padding-left, keyed to each button's own `key=` (Streamlit turns that into
# a `st-key-<sanitized key>` class on the button's wrapper -- see
# _nav_button_css_class), indents correctly regardless of label length.
_NAV_INDENT_BASE_PX = 8
_NAV_INDENT_STEP_PX = 18


def _nav_button_css_class(key: str) -> str:
    """The CSS class Streamlit's own frontend derives from a widget's
    `key=` ("if key is provided, it will be used as a CSS class name
    prefixed with st-key-"): the key, trimmed, with every character outside
    [a-zA-Z0-9_-] replaced by '-'."""
    return "st-key-" + re.sub(r"[^a-zA-Z0-9_-]", "-", key.strip())


def _render_navigation_node(key_prefix: str, is_root: bool, depth: int, css_rules: list[str]) -> None:
    """One sidebar button for the node at `key_prefix` -- clicking it
    switches the central column to show that node (see _select_node) --
    plus one more for each of its own children, recursively, each indented
    `_NAV_INDENT_STEP_PX` further than its parent (root sits at `depth` 0).
    Appends this node's own indentation rule to `css_rules`; _render_navigation
    joins the whole list into one <style> block once the full tree has been
    rendered. The currently selected node's own button renders as
    `type="primary"` so it stands out from the rest of the tree."""
    is_selected = _selected_node() == key_prefix
    button_key = f"nav::{key_prefix}"
    css_rules.append(
        f".{_nav_button_css_class(button_key)} button "
        f"{{ justify-content: flex-start; padding-left: {_NAV_INDENT_BASE_PX + depth * _NAV_INDENT_STEP_PX}px; }}"
    )
    st.sidebar.button(
        _nav_label(key_prefix, is_root), key=button_key,
        type="primary" if is_selected else "secondary",
        on_click=_select_node, args=(key_prefix,), use_container_width=True,
    )
    for child_id in _substrategy_children(key_prefix):
        _render_navigation_node(_child_key_prefix(key_prefix, child_id), False, depth + 1, css_rules)


def _render_navigation() -> None:
    """The sidebar's nested outline of the current sub-strategy tree,
    doubling as the switcher that controls which single node
    _render_substrategy shows in the central column -- rendered after (in
    code order) _render_substrategy has already run this script pass, so
    every node's own session_state (for its shorthand label, and for
    which node is currently selected) is already fresh."""
    st.sidebar.header("Navigation")
    st.sidebar.caption("Click a sub-strategy to view it.")
    css_rules: list[str] = []
    _render_navigation_node("root::", True, 0, css_rules)
    st.sidebar.html(f"<style>{''.join(css_rules)}</style>")


def main() -> None:
    st.set_page_config(page_title="CFR Strategy Explorer", layout="wide")
    st.title("CFR Strategy Explorer")

    checkpoint_path = st.sidebar.text_input("Checkpoint path", value=_default_checkpoint_path())
    max_samples = st.sidebar.number_input(
        "Max reservoir samples to load", min_value=100,
        value=DEFAULT_MAX_SAMPLES, step=1000, key="max_samples",
        help="How many samples to pull from the reservoir into memory, up front, for the whole session -- "
        "a big pool here is what gives an infrequent spot enough of its own matching rows to read a "
        "meaningful Decision variance explained figure, even though any *one* spot's own analysis is "
        "separately capped below.",
    )
    max_eval_samples = st.sidebar.number_input(
        "Max samples to evaluate in any given spot", min_value=100,
        value=DEFAULT_MAX_EVAL_SAMPLES, step=500, key="max_eval_samples",
        help="However many rows a sub-strategy's own claimed sample actually has, at most this many "
        "(a fixed random subsample, stable across reruns) are used for its own table/chart/metrics and "
        "every ranking or button here -- keeps a common, high-volume spot's own analysis fast regardless "
        "of how large the reservoir pool above is loaded, without capping how many *distinct* spots (rare "
        "ones included) that larger pool can still tell apart.",
    )

    if not (os.path.exists(f"{checkpoint_path}.pt") and os.path.exists(f"{checkpoint_path}.json")):
        st.error(f"No checkpoint found at {checkpoint_path}.{{pt,json}}")
        st.stop()

    _autoload_strategy_once(checkpoint_path)

    _net, net_config, _reservoir = _load_checkpoint(checkpoint_path)
    df = _build_dataframe(checkpoint_path, int(max_samples))
    if df.empty:
        st.warning("This checkpoint's reservoir is empty -- nothing to explore yet.")
        st.stop()

    st.caption(f"{len(df):,} reservoir samples loaded.")

    display_keys = cfr_features.display_feature_keys(net_config.feature_keys)

    st.sidebar.divider()
    _render_strategy_controls(checkpoint_path)

    st.sidebar.divider()
    collapsed = st.sidebar.toggle("Collapse actions to Fold / Call / Raise / All-In", value=False)

    # Only the currently selected sub-strategy renders its own content in
    # the central column -- every other node is reached by switching
    # selection instead of scrolling (see _render_substrategy/
    # _render_navigation's module docstring).
    _render_substrategy(df, _selected_node(), display_keys, collapsed, int(max_eval_samples))

    # Rendered after the selected node above so it reflects this run's
    # freshly updated session_state (filter picks, child list, selection)
    # -- see _render_navigation.
    _render_navigation()

    # Last, so it reflects this run's own final state -- every widget
    # interaction/callback above has already had its chance to mutate the
    # sub-strategy tree by this point (see _autosave_strategy).
    _autosave_strategy(checkpoint_path)


main()
