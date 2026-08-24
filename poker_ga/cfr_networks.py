"""The advantage network: an MLP (with residual connections between
same-width hidden layers -- see _ResidualBlock) regressing each of
strategy.ACTION_CATEGORIES' counterfactual regret, given the configured
feature subset (cfr_features.py) -- Single Deep CFR's only network (see
cfr_tree.py's module docstring for why there's just one, shared by every
seat, rather than a separate advantage net per player plus a second
strategy net).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import shap
import torch
import torch.nn as nn

import cfr_features
import strategy

DEFAULT_HIDDEN_SIZES = (2048, 2048, 2048, 2048, 2048, 2048, 2048)
DEFAULT_DROPOUT = 0.1


class _ResidualBlock(nn.Module):
    """One hidden layer -- Linear -> LayerNorm -> ReLU -> Dropout -- with a
    residual (skip) connection added around it whenever its input and
    output widths match: standard practice for deep MLPs, mirroring
    ResNet's identity shortcuts. Letting gradients flow straight through
    the addition keeps a deep stack (DEFAULT_HIDDEN_SIZES is 4 layers deep)
    trainable without the degradation a plain deep MLP is prone to, while
    leaving what the network can represent unchanged -- the block can
    still learn to zero out its own branch and pass its input through as
    though it weren't there at all. Skipped when in_dim != out_dim (only
    ever the very first hidden layer with DEFAULT_HIDDEN_SIZES, since every
    entry there is the same width) -- there's no dimension-matching
    identity to add in that case, and a learned projection shortcut isn't
    worth the added complexity for a single misaligned layer."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.same_width = in_dim == out_dim
        layers: list[nn.Module] = [nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.ReLU()]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        return x + out if self.same_width else out


class AdvantageNet(nn.Module):
    def __init__(
        self, input_dim: int, hidden_sizes: tuple[int, ...] = DEFAULT_HIDDEN_SIZES,
        output_dim: int = strategy.NUM_ACTION_CATEGORIES, dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_sizes = tuple(hidden_sizes)
        self.output_dim = output_dim
        self.dropout = dropout

        # Each hidden block is Linear -> LayerNorm -> ReLU -> Dropout, plus
        # a residual connection whenever widths line up -- see
        # _ResidualBlock. LayerNorm (not BatchNorm) since `predict` runs
        # single-sample (batch size 1) forward passes during CFR traversal
        # -- LayerNorm normalizes each sample independently and behaves
        # identically in train/eval, unlike BatchNorm, which needs a batch
        # to estimate statistics from and would otherwise need separate
        # handling for single-sample inference. Dropout is a no-op once
        # `predict`/`mean_shap_contributions_for_samples` call
        # `self.eval()`, so it only ever regularizes the minibatch
        # training steps in cfr_train.py's _train_step.
        prev = input_dim
        blocks: list[_ResidualBlock] = []
        for h in hidden_sizes:
            blocks.append(_ResidualBlock(prev, h, dropout))
            prev = h
        self.hidden = nn.ModuleList(blocks)
        self.output_layer = nn.Linear(prev, output_dim)  # linear output: this is a regression head, not a classifier
        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming-normal init (fan_in, ReLU gain) for every hidden block's
        own Linear layer, matching the ReLU nonlinearity that follows each
        -- PyTorch's own default Linear init assumes a Leaky ReLU-ish gain
        that's a reasonable general default but not tuned to this
        network's actual activation. The final output Linear (a
        regression head, no activation after it) instead gets a
        small-gain Xavier init, so the net's initial regret predictions
        start out close to 0 rather than with ReLU-scaled variance that
        has no reason to match the true target scale."""
        for block in self.hidden:
            linear = block.block[0]
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
            nn.init.zeros_(linear.bias)
        nn.init.xavier_normal_(self.output_layer.weight, gain=0.1)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.hidden:
            x = block(x)
        return self.output_layer(x)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """features: shape (input_dim,). Returns raw regret estimates,
        shape (output_dim,) -- the interface cfr_tree.py's traversal
        expects from a `net`."""
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(features).unsqueeze(0)
            out = self.forward(x).squeeze(0)
        return out.numpy()


def clone(net: AdvantageNet) -> AdvantageNet:
    """An independent copy of `net`'s weights -- e.g. a snapshot taken
    right before a training update, so it can still be played against
    (and, if that update didn't help, reverted to) after the original has
    since moved on. `load_state_dict` copies tensor *values* into the new
    module's own parameters rather than aliasing the source's, so later
    training on `net` can never leak into the clone."""
    cloned = AdvantageNet(
        input_dim=net.input_dim, hidden_sizes=net.hidden_sizes, output_dim=net.output_dim, dropout=net.dropout,
    )
    cloned.load_state_dict(net.state_dict())
    cloned.eval()
    return cloned


@dataclass
class AdvantageNetConfig:
    feature_keys: tuple[str, ...]
    hidden_sizes: tuple[int, ...]
    table_size: int
    dropout: float = DEFAULT_DROPOUT


def save(net: AdvantageNet, config: AdvantageNetConfig, path: str) -> None:
    """Writes `<path>.pt` (weights) and `<path>.json` (a self-describing
    sidecar: feature keys, architecture, table size) -- mirrors genome.py's
    save-what-you-need-to-reconstruct-with philosophy."""
    torch.save(net.state_dict(), f"{path}.pt")
    with open(f"{path}.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_keys": list(config.feature_keys),
                "hidden_sizes": list(config.hidden_sizes),
                "table_size": config.table_size,
                "dropout": config.dropout,
                "action_categories": strategy.ACTION_CATEGORIES,
            },
            f,
        )


def _normalized_mean_abs_shap(shap_values: list[np.ndarray]) -> np.ndarray:
    """`shap_values` as shap.Explainer.shap_values returns it -- one
    (n_samples, n_features) array per action category -- reduced to one
    mean-|SHAP| score per feature.

    Each (action, feature) pair is re-centered on its own mean across
    `n_samples` before taking the mean absolute value: a feature that
    contributes the same constant amount (e.g. +20) to every one of those
    samples isn't distinguishing between any of them, so it should score
    ~0 despite a large raw |SHAP| -- exactly the case where a feature
    looks important over a whole reservoir but turns out to be constant
    (and so, in this reduced view, uninformative) within one particular
    filtered subset of it (see cfr_explorer.py). Applied unconditionally,
    not just for a caller that happens to be filtering, since a constant
    contribution is uninformative regardless of how the samples were
    chosen -- pulled out of mean_shap_contributions_for_samples as a pure
    function so this normalization step can be unit-tested against
    synthetic arrays, independent of shap's own sampling noise."""
    raw = np.stack(shap_values, axis=0)  # (NUM_ACTIONS, n_samples, n_features)
    centered = raw - raw.mean(axis=1, keepdims=True)  # zero out each (action, feature)'s constant offset across samples
    return np.mean(np.abs(centered), axis=(0, 1))


def _validity_codes(valid: np.ndarray) -> np.ndarray:
    """One integer per row of `valid` (see cfr_features.unmasked_validity),
    bit-packing which columns are real (1) vs masked (0) -- two rows get the
    same code exactly when they share the same masking pattern across every
    feature (e.g., for this codebase's actual masking conditions, every row
    from the same street)."""
    weights = (1 << np.arange(valid.shape[1], dtype=np.int64))
    return valid.astype(np.int64) @ weights


def mean_shap_contributions_for_samples(
    net: AdvantageNet, explain_features: np.ndarray, background_features: np.ndarray,
    feature_keys: tuple[str, ...], rng: np.random.Generator,
    sample_size: int = 200, background_size: int = 20, nsamples: int = 20,
    explain_group_labels: np.ndarray | None = None, background_group_labels: np.ndarray | None = None,
) -> list[tuple[str, float]]:
    """Normalized mean |SHAP value| per feature (see _normalized_mean_abs_shap;
    shap.GradientExplainer's Expected Gradients approximation -- exact
    Shapley values are NP-hard, this is what SHAP itself uses for neural
    nets too), averaged over every one of the net's NUM_ACTION_CATEGORIES
    outputs.

    Takes an explicit pool of rows to explain (rather than reading straight
    from a reservoir) so a caller like cfr_explorer.py can restrict to just
    the rows that pass the sidebar's current filters. `sample_size`/
    `background_size` bound the cost to roughly constant regardless of pool
    size.

    `explain_group_labels`/`background_group_labels`, if given, have one
    entry per row of `explain_features`/`background_features` respectively
    -- e.g. a sub-strategy's own *parent's* resolved Split By group (see
    cfr_explorer._group_labels_for_rows). The two pools don't need to be
    the same rows, or even the same size: cfr_explorer._render_substrategy
    deliberately passes a *wider* background pool (the parent's own full
    claimed scope) than the (narrower, e.g. a child sub-strategy's own
    claimed) explain pool, specifically so a group's baseline reflects how
    the parent's grouping feature behaves in general, not just within
    whatever narrow slice is being explained -- see that function's own
    docstring for why (in short: a child sub-strategy applying the exact
    same Split By feature as its parent, over a small subset, can still
    capture real, additional signal the parent's own broader-population
    analysis of that same feature doesn't -- the two shouldn't be forced
    to agree). `background_group_labels` defaults to `explain_group_labels`
    when omitted, matching the self-referential explain=background case
    (e.g. this module's own tests).

    Group labels stratify on top of (not instead of) the masking-pattern
    stratification below: each combined (masking pattern, group) stratum
    is explained using ONLY a background sharing that exact combination.
    When background is drawn from the *same* population as explain (the
    self-referential case), a feature the grouping fully, structurally
    determines (e.g. Hole Suited, fully determined by Exact Hole Hand) is
    an exactly-constant reading for the whole stratum, background and
    explain alike, so x_i - background_i is exactly 0 at every
    interpolation point and it scores exactly 0. When background is drawn
    from a *wider* population instead, that same feature only scores near
    0 if its relationship to the target is *also* uniform across that
    wider population -- a relationship that's genuinely different (or
    stronger, or reversed) specifically within the narrower explain pool
    shows up as a nonzero residual instead, exactly the "does this narrower
    rule still add something" question a sub-strategy's own Split By
    metric is meant to answer.

    Computed separately per distinct masking pattern (see
    cfr_features.unmasked_validity/_validity_codes -- in practice, one
    stratum per street, since that's what actually drives every `maskable`
    feature's masking condition in this codebase, though nothing here
    hardcodes "street"), each explained using ONLY a same-pattern-masked
    background, then combined via a weighted average (weight = that
    stratum's own share of the sampled explain rows). A single joint
    computation over the whole (mixed) pool would let a masked feature get
    explained against an unmasked background (or vice versa) -- since every
    `maskable` feature that shares one masking condition (e.g. every
    draw-shape feature, all masked together at the river) is then perfectly
    collinear with whatever real effect actually distinguishes that
    condition (e.g. street_norm's own river bucket), a gradient-based
    explainer can misattribute that real effect onto the masked features
    instead of the feature actually responsible for it -- confirmed with a
    synthetic experiment where masked features with *zero* true causal
    effect still outranked the real cause of an effect under the mixed-pool
    computation. Within one masking-pattern stratum, every masked feature is
    an exactly-constant reading for the whole stratum (background and
    explain alike), so it contributes exactly 0 there without needing any
    special-casing -- the real fix is just never letting background and
    explain disagree about which features are masked.

    Returns (feature_key, mean_abs_shap) pairs sorted most-to-least
    important. Empty list if `explain_features` is empty."""
    if len(explain_features) == 0:
        return []
    if background_group_labels is None:
        background_group_labels = explain_group_labels

    explain_codes = _validity_codes(cfr_features.unmasked_validity(feature_keys, explain_features))
    background_codes = _validity_codes(cfr_features.unmasked_validity(feature_keys, background_features))

    if explain_group_labels is None:
        explain_strata = explain_codes
        background_strata = background_codes
    else:
        # Combining as strings rather than e.g. packing into a single int
        # keeps this agnostic to whatever type group labels happen to be
        # (str for an ordinary bucket-label grouping, int for Exact Hole
        # Hand's grid-cell grouping -- see cfr_explorer._group_labels_for_rows).
        explain_strata = np.char.add(np.char.add(explain_codes.astype(str), ":"), explain_group_labels.astype(str))
        background_strata = np.char.add(np.char.add(background_codes.astype(str), ":"), background_group_labels.astype(str))

    net.eval()
    totals = np.zeros(len(feature_keys))
    total_weight = 0
    for stratum in np.unique(explain_strata):
        stratum_explain_idx = np.flatnonzero(explain_strata == stratum)
        stratum_background_idx = np.flatnonzero(background_strata == stratum)
        if len(stratum_background_idx) == 0:
            # No same-stratum background available -- skip rather than
            # fall back to a background that would reintroduce the exact
            # mismatch this is meant to avoid.
            continue

        # Each stratum's own share of sample_size, proportional to its
        # share of the explain pool, so the total explain rows used across
        # every stratum stays close to sample_size regardless of how many
        # distinct strata are present.
        stratum_n = max(1, round(sample_size * len(stratum_explain_idx) / len(explain_strata)))
        explain_n = min(stratum_n, len(stratum_explain_idx))
        explain_idx = rng.choice(stratum_explain_idx, size=explain_n, replace=False)
        x = torch.from_numpy(explain_features[explain_idx])

        background_n = min(background_size, len(stratum_background_idx))
        background_idx = rng.choice(stratum_background_idx, size=background_n, replace=False)
        background = torch.from_numpy(background_features[background_idx])

        explainer = shap.GradientExplainer(net, background)
        shap_values = explainer.shap_values(x, nsamples=nsamples)  # list of NUM_ACTIONS arrays, each (explain_n, num_features)
        totals += _normalized_mean_abs_shap(shap_values) * explain_n
        total_weight += explain_n

    if total_weight == 0:
        return []
    mean_abs = totals / total_weight

    order = np.argsort(-mean_abs)
    return [(feature_keys[i], float(mean_abs[i])) for i in order]


def load(path: str) -> tuple[AdvantageNet, AdvantageNetConfig]:
    with open(f"{path}.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_keys = tuple(meta["feature_keys"])
    hidden_sizes = tuple(meta["hidden_sizes"])
    # .get, not meta[...]: a checkpoint saved before dropout existed simply
    # didn't have any (an unregularized net is equivalent to dropout=0.0),
    # not an error -- same "missing means it predates the field" treatment
    # Trainer.load_trainer_state gives an absent trainer-state file.
    dropout = meta.get("dropout", 0.0)
    input_dim = len(cfr_features.feature_indices(feature_keys))

    net = AdvantageNet(input_dim=input_dim, hidden_sizes=hidden_sizes, dropout=dropout)
    net.load_state_dict(torch.load(f"{path}.pt", map_location="cpu"))
    net.eval()

    config = AdvantageNetConfig(
        feature_keys=feature_keys, hidden_sizes=hidden_sizes, table_size=meta["table_size"], dropout=dropout,
    )
    return net, config
