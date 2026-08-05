"""The advantage network: a plain MLP regressing each of
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

DEFAULT_HIDDEN_SIZES = (512, 512, 512)
DEFAULT_DROPOUT = 0.1


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

        # Each hidden block is Linear -> LayerNorm -> ReLU -> Dropout.
        # LayerNorm (not BatchNorm) since `predict` runs single-sample
        # (batch size 1) forward passes during CFR traversal -- LayerNorm
        # normalizes each sample independently and behaves identically in
        # train/eval, unlike BatchNorm, which needs a batch to estimate
        # statistics from and would otherwise need separate handling for
        # single-sample inference. Dropout is a no-op once `predict`/
        # `mean_shap_contributions_for_samples` call `self.eval()`, so it
        # only ever regularizes the minibatch training steps in
        # cfr_train.py's _train_step.
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))  # linear output: this is a regression head, not a classifier
        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming-normal init (fan_in, ReLU gain) for every hidden Linear
        layer, matching the ReLU nonlinearity that follows each -- PyTorch's
        own default Linear init assumes a Leaky ReLU-ish gain that's a
        reasonable general default but not tuned to this network's actual
        activation. The final output Linear (a regression head, no
        activation after it) instead gets a small-gain Xavier init, so the
        net's initial regret predictions start out close to 0 rather than
        with ReLU-scaled variance that has no reason to match the true
        target scale."""
        hidden_linears = [m for m in self.model if isinstance(m, nn.Linear)][:-1]
        for layer in hidden_linears:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        output_layer = self.model[-1]
        nn.init.xavier_normal_(output_layer.weight, gain=0.1)
        nn.init.zeros_(output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

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


def mean_shap_contributions_for_samples(
    net: AdvantageNet, explain_features: np.ndarray, background_features: np.ndarray,
    feature_keys: tuple[str, ...], rng: np.random.Generator,
    sample_size: int = 200, background_size: int = 20, nsamples: int = 20,
) -> list[tuple[str, float]]:
    """Normalized mean |SHAP value| per feature (see _normalized_mean_abs_shap;
    shap.GradientExplainer's Expected Gradients approximation -- exact
    Shapley values are NP-hard, this is what SHAP itself uses for neural
    nets too) over a random subsample of `explain_features`, using a
    random subsample of `background_features` as the explainer's reference
    distribution, averaged over every one of the net's
    NUM_ACTION_CATEGORIES outputs.

    Takes an explicit pool of rows to explain (rather than reading straight
    from a reservoir) so a caller like cfr_explorer.py can restrict to just
    the rows that pass the sidebar's current filters. `sample_size`/
    `background_size` bound the cost to roughly constant regardless of pool
    size.

    Returns (feature_key, mean_abs_shap) pairs sorted most-to-least
    important. Empty list if `explain_features` is empty."""
    if len(explain_features) == 0:
        return []

    explain_n = min(sample_size, len(explain_features))
    explain_idx = rng.choice(len(explain_features), size=explain_n, replace=False)
    x = torch.from_numpy(explain_features[explain_idx])

    background_n = min(background_size, len(background_features))
    background_idx = rng.choice(len(background_features), size=background_n, replace=False)
    background = torch.from_numpy(background_features[background_idx])

    net.eval()
    explainer = shap.GradientExplainer(net, background)
    shap_values = explainer.shap_values(x, nsamples=nsamples)  # list of NUM_ACTIONS arrays, each (explain_n, num_features)
    mean_abs = _normalized_mean_abs_shap(shap_values)

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
