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


@st.cache_resource(show_spinner="Ranking features by SHAP contribution...")
def _feature_importance(checkpoint_path: str) -> list[tuple[str, float]]:
    net, net_config, reservoir = _load_checkpoint(checkpoint_path)
    return cfr_networks.mean_shap_contributions(net, reservoir, net_config.feature_keys, np.random.default_rng(0))


@st.cache_resource(show_spinner="Computing the current strategy over reservoir samples...")
def _build_dataframe(checkpoint_path: str, max_samples: int) -> pd.DataFrame:
    """One row per (subsampled) reservoir entry: one Categorical column per
    feature (its bucket label, in the feature's own natural value order --
    see cfr_features.bucket_categories) and one float column per action
    category (the *current* net's regret-matching probability for that
    action, given that row's own legal-action mask). Everything the UI does
    afterward is just pandas filtering/grouping over this, computed once."""
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

    data = {}
    for i, key in enumerate(net_config.feature_keys):
        labels = cfr_features.bucket_labels(key, features[:, i])
        data[_FEATURE_COL_PREFIX + key] = pd.Categorical(
            labels, categories=cfr_features.bucket_categories(key), ordered=True,
        )
    for i, label in enumerate(strategy.ACTION_CATEGORIES):
        data[_ACTION_COL_PREFIX + label] = probs[:, i]

    return pd.DataFrame(data)


def _feature_col(key: str) -> str:
    return _FEATURE_COL_PREFIX + key


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
    st.sidebar.caption("Ordered by mean |SHAP| contribution to the net's predictions.")

    filters: dict[str, list[str]] = {}
    group_split_keys: list[str] = []
    table_split_keys: list[str] = []

    for key, importance in feature_order:
        col = _feature_col(key)
        role = st.sidebar.selectbox(f"{key}  (SHAP {importance:.4f})", ROLES, key=f"role::{key}")
        if role == ROLE_FILTER:
            observed = [c for c in cfr_features.bucket_categories(key) if c in set(df[col])]
            filters[key] = st.sidebar.multiselect("keep values", observed, default=observed, key=f"filter::{key}")
        elif role == ROLE_GROUP_SPLIT:
            group_split_keys.append(key)
        elif role == ROLE_TABLE_SPLIT:
            table_split_keys.append(key)

    if len(table_split_keys) > MAX_TABLE_SPLIT_FEATURES:
        kept, dropped = table_split_keys[:MAX_TABLE_SPLIT_FEATURES], table_split_keys[MAX_TABLE_SPLIT_FEATURES:]
        st.error(
            f"Only {MAX_TABLE_SPLIT_FEATURES} features can be used as table splits at once -- "
            f"using {', '.join(kept)} (by SHAP rank) and ignoring {', '.join(dropped)}. Change one "
            "of those features' roles to use a different pair."
        )
        table_split_keys = kept

    st.sidebar.divider()
    collapsed = st.sidebar.toggle("Collapse actions to Fold / Call / Raise / All-In", value=False)

    return filters, group_split_keys, table_split_keys, collapsed


def _apply_filters(df: pd.DataFrame, filters: dict[str, list[str]]) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for key, values in filters.items():
        mask &= df[_feature_col(key)].isin(values)
    return df[mask]


def _render_table(group_df: pd.DataFrame, table_split_keys: list[str], collapsed: bool) -> None:
    action_cols = _action_columns(collapsed)
    view = _with_action_view(group_df, collapsed)

    if not table_split_keys:
        summary = view[action_cols].mean().to_frame("All rows").T
        counts = pd.DataFrame({"n": [len(view)]}, index=["All rows"])
    elif len(table_split_keys) == 1:
        idx_col = _feature_col(table_split_keys[0])
        summary = view.groupby(idx_col, observed=True)[action_cols].mean()
        counts = view.groupby(idx_col, observed=True).size().to_frame("n")
    else:
        row_col, col_col = _feature_col(table_split_keys[0]), _feature_col(table_split_keys[1])
        summary = pd.pivot_table(view, values=action_cols, index=row_col, columns=col_col, aggfunc="mean", observed=True)
        summary = summary.swaplevel(axis=1).sort_index(axis=1, level=0)
        counts = pd.pivot_table(view, values=action_cols[0], index=row_col, columns=col_col, aggfunc="count", observed=True)

    st.dataframe((summary * 100).round(1).astype(str) + "%")
    with st.expander(f"Sample counts (total n={len(view):,})"):
        st.dataframe(counts)


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

    feature_importance = _feature_importance(checkpoint_path)
    df = _build_dataframe(checkpoint_path, int(max_samples))
    if df.empty:
        st.warning("This checkpoint's reservoir is empty -- nothing to explore yet.")
        st.stop()

    st.caption(f"{len(df):,} reservoir samples loaded.")

    filters, group_split_keys, table_split_keys, collapsed = _render_sidebar(feature_importance, df)
    filtered = _apply_filters(df, filters)

    if filtered.empty:
        st.warning("No reservoir samples match the current filters.")
        st.stop()

    if not group_split_keys:
        _render_table(filtered, table_split_keys, collapsed)
    else:
        group_cols = [_feature_col(k) for k in group_split_keys]
        for group_values, group_df in filtered.groupby(group_cols, observed=True):
            if len(group_df) == 0:
                continue
            if not isinstance(group_values, tuple):
                group_values = (group_values,)
            header = ", ".join(f"{k} = {v}" for k, v in zip(group_split_keys, group_values))
            st.subheader(f"{header}  (n={len(group_df):,})")
            _render_table(group_df, table_split_keys, collapsed)


main()
