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
    background_digest: str, _background_row_index: np.ndarray, group_by_keys: tuple[str, ...] = (),
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

    `_background_row_index` is ordinarily the same as `_row_index`
    (self-referential -- explains and backgrounds against its own rows),
    but callers pass a *wider* pool -- see _render_substrategy -- when
    `group_by_keys` is set, so a parent's grouping gets corrected for
    using the parent's own broader sample rather than this (possibly much
    narrower) sub-strategy's own rows -- see cfr_networks.
    mean_shap_contributions_for_samples for why that distinction matters.

    `group_by_keys` (a parent sub-strategy's own resolved Split By pair --
    see _render_substrategy) makes this sub-strategy's own view assume that
    parent grouping is already "priced in": each row's contribution is
    centered against its own group's mean, computed over
    `_background_row_index` (see _group_labels_for_rows/cfr_networks.
    _normalized_mean_abs_shap), instead of over its whole row set's own
    mean, so a feature the parent's grouping already fully (and uniformly,
    across the parent's whole sample) explains scores ~0 here too, rather
    than getting credited again for signal the parent's own grouping
    already accounts for."""
    net, net_config, _reservoir = _load_checkpoint(checkpoint_path)
    display_keys = cfr_features.display_feature_keys(net_config.feature_keys)
    if len(_row_index) == 0:
        return [(key, 0.0) for key in display_keys]
    df, raw_features = _build_dataframe(checkpoint_path, max_samples)
    selected = raw_features[_row_index]
    background = raw_features[_background_row_index]
    explain_group_labels = _group_labels_for_rows(df, _row_index, group_by_keys)
    background_group_labels = _group_labels_for_rows(df, _background_row_index, group_by_keys)
    contributions = cfr_networks.mean_shap_contributions_for_samples(
        net, selected, background, net_config.feature_keys, np.random.default_rng(0),
        explain_group_labels=explain_group_labels, background_group_labels=background_group_labels,
    )
    folded = dict(cfr_features.fold_child_contributions(contributions))
    return sorted(((key, folded.get(key, 0.0)) for key in display_keys), key=lambda kv: -kv[1])


@st.cache_data(show_spinner="Ranking cross-features by interaction strength...")
def _interaction_strength_for_key(
    checkpoint_path: str, max_samples: int, row_digest: str, _row_index: np.ndarray, focus_key: str,
) -> list[tuple[str, float]]:
    """Mean absolute pairwise interaction effect (see
    cfr_networks.interaction_strength_for_feature) between `focus_key` and
    every other net-input feature, over the rows at `_row_index` (a graphed
    feature's own current `filtered` pool -- see _render_graphs). Unlike
    _shap_importance_for_rows, this needs no parent-grouping correction: the
    underlying Delta_ij term already isolates the interaction between
    exactly two features, unaffected by any other feature's (however
    dominant) own additive effect -- see that function's own docstring.
    Self-referential (background = the same pool as explain), same as
    _shap_importance_for_rows' own root-node case."""
    net, net_config, _reservoir = _load_checkpoint(checkpoint_path)
    if len(_row_index) == 0:
        return [(key, 0.0) for key in net_config.feature_keys if key != focus_key]
    _df, raw_features = _build_dataframe(checkpoint_path, max_samples)
    pool = raw_features[_row_index]
    return cfr_networks.interaction_strength_for_feature(
        net, pool, pool, net_config.feature_keys, focus_key, np.random.default_rng(0),
    )


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
    filtered: pd.DataFrame, graph_keys: list[str], display_keys: list[str], collapsed: bool,
    checkpoint_path: str, max_samples: int, key_prefix: str = "",
) -> None:
    """One line chart per feature picked via a sub-strategy's own local
    "Add graph" control (see _render_substrategy), each paired with a
    multiselect of other features to "cross" it with, ranked strongest-
    interaction-first and labeled with that strength (see
    cfr_networks.interaction_strength_for_feature/_interaction_strength_for_key)
    -- every feature picked there adds its own row of 4 heatmaps (this
    graphed feature as the x axis, the picked feature as the y axis, one
    heatmap per simplified action rate). Exact Hole Hand (_HOLE_HAND_GRID_KEY)
    is the one exception: inherently 2D already, it renders its own fixed set of
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
        row_index = filtered.index.to_numpy()
        interaction = _interaction_strength_for_key(
            checkpoint_path, max_samples, _row_digest(row_index), row_index, key,
        )
        interaction_by_key = dict(interaction)
        # Strongest-interaction-first, restricted to (and keeping) every
        # option other_keys itself offers -- interaction's own order
        # otherwise ranges over every net-input feature (incl. ones not
        # offered here, e.g. linked children -- see cfr_networks.
        # interaction_strength_for_feature).
        ranked_other_keys = [k for k, _ in interaction if k in other_keys]

        def _cross_option_label(k: str) -> str:
            return f"{cfr_features.feature_label(k)}  (Interaction {interaction_by_key.get(k, 0.0):.4f})"

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


def _decision_variance_explained(
    df: pd.DataFrame, default_df: pd.DataFrame, parent_node_df: pd.DataFrame | None,
    parent_group_by_keys: tuple[str, ...], resolved_split_by: list[str], collapsed: bool,
) -> float | None:
    """"Decision variance explained by Split By features on claimed
    samples" (see _render_substrategy) -- the fraction of variance in the
    net's own predicted action-probability vectors, across this node's
    own claimed-and-not-further-claimed rows (`default_df`), that grouping
    those rows by `resolved_split_by` (this node's own chosen Split By
    feature(s)) explains -- a standard ANOVA/eta-squared "variance
    explained by grouping" statistic, computed directly on the model's own
    decisions rather than via SHAP attribution. SHAP can't cleanly answer
    this particular question: _shap_importance_for_rows corrects for a
    parent's own grouping by restricting SHAP's own background to rows
    sharing the same group value, which forces a feature's own direct
    attribution toward 0 whenever it *is* the grouping feature itself,
    structurally, regardless of whether reusing it here captures real,
    additional signal -- exactly the "shows 0% even though a different,
    narrower rule is genuinely being applied" bug this function exists to
    avoid. None if `resolved_split_by` is empty (nothing chosen yet to
    measure).

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
    ordinary, unadjusted ANOVA."""
    if not resolved_split_by:
        return None
    action_cols = _action_columns(collapsed)
    default_view = _with_action_view(default_df, collapsed)
    default_index = default_df.index.to_numpy()
    default_actions = default_view[action_cols].to_numpy()

    if parent_group_by_keys and parent_node_df is not None and not parent_node_df.empty:
        parent_view = _with_action_view(parent_node_df, collapsed)
        parent_labels = _group_labels_for_rows(df, parent_node_df.index.to_numpy(), parent_group_by_keys)
        parent_group_means = pd.DataFrame(parent_view[action_cols].to_numpy(), index=parent_labels).groupby(level=0).mean()
        own_parent_labels = _group_labels_for_rows(df, default_index, parent_group_by_keys)
        baseline = parent_group_means.reindex(own_parent_labels).to_numpy()
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
    if total_variance <= 0.0:
        return 0.0

    own_labels = _group_labels_for_rows(df, default_index, tuple(resolved_split_by))
    own_group_means = pd.DataFrame(residual, index=own_labels).groupby(level=0).transform("mean").to_numpy()
    within_group_variance = float(((residual - own_group_means) ** 2).sum(axis=1).mean())

    return max(0.0, 1.0 - within_group_variance / total_variance) * 100


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
    full claimed scope: the background pool a sub-strategy's own SHAP
    computation should draw from when correcting for a grouping the
    parent's Split By already applies (see _render_substrategy) --
    deliberately the parent's full node_df, not its narrower default_df
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


def _render_substrategy(
    root_df: pd.DataFrame, key_prefix: str, checkpoint_path: str, max_samples: int,
    display_keys: list[str], collapsed: bool,
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
    to nothing), the prominent table+graph, and the "Decision variance
    explained by Split By features on claimed samples" metric, all
    computed over that leftover, since the prominent block *is* the
    default/else branch of the rule list -- then purely illustrative
    "Add table"/"Add graph" additions, also over that same leftover.

    Both SHAP views this node computes (`importance`, over its own
    incoming pool, for "Add filter"; `default_importance`, over its own
    claimed-and-not-further-claimed leftover, for Split By/the variance
    metric) correct for whatever grouping this node's own *parent*
    already applies (`parent_group_by_keys`) by centering against
    `parent_node_df` -- the parent's own full claimed scope (see
    _resolve_node_df), not this node's own rows. That distinction matters:
    if the correction used this node's own (possibly much narrower) rows
    instead, a feature merely *constant* within that narrow slice would
    always score 0 regardless of whether the parent's grouping actually
    explains it there -- e.g. a child sharing its parent's own Split By
    feature, filtered down to a small subset, would always show 0%
    "explained" even though that narrower rule can genuinely capture real,
    additional signal the parent's broader-population analysis doesn't.
    Centering against the parent's own full sample instead means a
    feature only reads as "already explained" when the parent's grouping
    is *uniformly* true across the parent's whole domain, not just within
    whatever this node happens to have claimed."""
    parent_key_prefix, child_id = _parent_key_prefix_and_child_id(key_prefix)
    is_root = parent_key_prefix is None
    incoming_df = _resolve_incoming_df(root_df, key_prefix)

    # This node's own SHAP views (both calls below) assume its *parent's*
    # own Split By grouping is already "priced in" -- see
    # _group_labels_for_rows/cfr_networks._normalized_mean_abs_shap. The
    # background pool for that correction is the *parent's* own full
    # claimed scope (see _resolve_node_df), not this node's own (possibly
    # much narrower) rows -- a feature the parent's grouping only
    # partially/locally explains should still show up as informative here,
    # not collapse to 0 just because it's constant within this node's own
    # narrow sample.
    parent_split_by = [] if is_root else _split_by_from_state(parent_key_prefix)
    parent_group_by_keys = tuple(_resolve_split_by(parent_split_by))
    parent_node_df = None if is_root else _resolve_node_df(root_df, parent_key_prefix)

    def _background_index(explain_index: np.ndarray) -> np.ndarray:
        if parent_group_by_keys and parent_node_df is not None:
            return parent_node_df.index.to_numpy()
        return explain_index

    row_index = incoming_df.index.to_numpy()
    background_index = _background_index(row_index)
    importance = _shap_importance_for_rows(
        checkpoint_path, max_samples, _row_digest(row_index), row_index,
        _row_digest(background_index), background_index, group_by_keys=parent_group_by_keys,
    )
    non_graph_options = [k for k, _ in importance if k != _HOLE_HAND_GRID_KEY]
    importance_by_key = dict(importance)

    def _claim_option_label(key: str) -> str:
        return f"{cfr_features.feature_label(key)}  (SHAP {importance_by_key[key]:.4f})"

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

    claim_desc = ", ".join(
        f"{cfr_features.feature_label(k)} = {', '.join(v)}" for k, v in local_filters.items()
    )
    if is_root:
        heading = f"Overall Strategy ({claim_desc})" if claim_desc else "Overall Strategy"
        st.header(f"{heading}  (n={len(node_df):,})")
    else:
        heading = claim_desc or "Sub-strategy (no filter set yet -- claims everything its parent hasn't)"
        col_heading, col_up, col_down, col_remove = st.columns([10, 1, 1, 1])
        with col_heading:
            st.header(f"{heading}  (n={len(node_df):,})")
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

    default_df = _resolve_default_df(node_df, key_prefix)
    default_row_index = default_df.index.to_numpy()
    default_background_index = _background_index(default_row_index)
    default_importance = _shap_importance_for_rows(
        checkpoint_path, max_samples, _row_digest(default_row_index), default_row_index,
        _row_digest(default_background_index), default_background_index, group_by_keys=parent_group_by_keys,
    )
    importance_by_key = dict(default_importance)

    def _default_option_label(key: str) -> str:
        return f"{cfr_features.feature_label(key)}  (SHAP {importance_by_key[key]:.4f})"

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
                root_df, default_df, parent_node_df, parent_group_by_keys, resolved_split_by, collapsed,
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
    _render_graphs(default_df, extra_graph_keys, display_keys, collapsed, checkpoint_path, max_samples, f"{key_prefix}extra::")


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


def _render_navigation_node(key_prefix: str, is_root: bool, depth: int) -> None:
    """One sidebar button for the node at `key_prefix` -- clicking it
    switches the central column to show that node (see _select_node) --
    plus one more, indented one level deeper, for each of its own
    children, recursively. The currently selected node's own button
    renders as `type="primary"` so it stands out from the rest of the
    tree."""
    is_selected = _selected_node() == key_prefix
    indent = " " * (depth * 4)  # non-breaking spaces -- regular spaces collapse in rendered HTML
    st.sidebar.button(
        f"{indent}{_nav_label(key_prefix, is_root)}", key=f"nav::{key_prefix}",
        type="primary" if is_selected else "secondary",
        on_click=_select_node, args=(key_prefix,), use_container_width=True,
    )
    for child_id in _substrategy_children(key_prefix):
        _render_navigation_node(_child_key_prefix(key_prefix, child_id), False, depth + 1)


def _render_navigation() -> None:
    """The sidebar's nested outline of the current sub-strategy tree,
    doubling as the switcher that controls which single node
    _render_substrategy shows in the central column -- rendered after (in
    code order) _render_substrategy has already run this script pass, so
    every node's own session_state (for its shorthand label, and for
    which node is currently selected) is already fresh."""
    st.sidebar.header("Navigation")
    st.sidebar.caption("Click a sub-strategy to view it.")
    _render_navigation_node("root::", True, 0)


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

    # Only the currently selected sub-strategy renders its own content in
    # the central column -- every other node is reached by switching
    # selection instead of scrolling (see _render_substrategy/
    # _render_navigation's module docstring).
    _render_substrategy(df, _selected_node(), checkpoint_path, int(max_samples), display_keys, collapsed)

    # Rendered after the selected node above so it reflects this run's
    # freshly updated session_state (filter picks, child list, selection)
    # -- see _render_navigation.
    _render_navigation()


main()
