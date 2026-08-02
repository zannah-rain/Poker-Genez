"""Outer Single Deep CFR training loop: alternates Monte Carlo tree
traversals (cfr_tree.py, filling the shared reservoir) with minibatch
regression steps on the shared advantage network, weighting each sample's
loss by its iteration number t (see _train_step) so later iterations
dominate the training signal -- that's what gives "Single" Deep CFR its
average-strategy approximation without a second policy network/reservoir:
the *final* net's regret-matching strategy already approximates the
average strategy, the same way Linear CFR's running average strategy is
dominated by its later iterations.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch

import cfr_features
import cfr_networks
import cfr_reservoir
import cfr_tree
import strategy
from game import GameConfig


@dataclass
class DeepCFRConfig:
    feature_keys: tuple[str, ...] = cfr_features.DEFAULT_FEATURE_KEYS
    hidden_sizes: tuple[int, ...] = cfr_networks.DEFAULT_HIDDEN_SIZES
    table_size: int = 6
    iterations: int = 100
    traversals_per_iteration: int = 200
    sgd_steps_per_iteration: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    reservoir_capacity: int = 200_000
    num_equity_rollouts: int = cfr_tree.DEFAULT_NUM_EQUITY_ROLLOUTS
    game_config: GameConfig = field(default_factory=GameConfig)


def _train_step(
    net: cfr_networks.AdvantageNet, optimizer: torch.optim.Optimizer,
    reservoir: cfr_reservoir.ReservoirBuffer, batch_size: int, current_iteration: int,
) -> float:
    features, regret_targets, legal_mask, raw_weights = reservoir.sample(batch_size)
    # Each sample's weight is the outer iteration t it was collected at
    # (reservoir.add's `t`), which is what gives Linear CFR its "recent
    # iterations matter more" shape -- but left unnormalized, that weight
    # grows without bound as training progresses (t climbing into the
    # hundreds), which does two bad things at once: the *reported* loss
    # trends upward run-over-run even when predictions aren't getting any
    # worse (t is dominating the number, not error), and -- worse -- the
    # actual gradient magnitude grows with it, since Adam's per-parameter
    # second-moment estimate (a decayed running average, so it lags a
    # steadily climbing gradient scale) increasingly underestimates the
    # true gradient magnitude and so takes increasingly large steps. Both
    # look exactly like "loss is diverging". Dividing by current_iteration
    # keeps every weight in (0, 1] -- a sample from *this* iteration always
    # gets weight 1.0, older ones fade toward 0 -- which preserves the same
    # relative recent-favoring shape without letting the absolute scale run
    # away over the course of a long training run.
    weights = raw_weights / current_iteration
    predicted = net(features)
    legal_mask_f = legal_mask.float()
    squared_error = (predicted - regret_targets) ** 2 * legal_mask_f
    # Mean squared error over this sample's *legal* actions only -- an
    # illegal action's regret target is meaningless (never had a value to
    # compare against), so it shouldn't pull the network's prediction there
    # toward 0.
    per_sample_loss = squared_error.sum(dim=1) / legal_mask_f.sum(dim=1).clamp(min=1.0)
    loss = (weights * per_sample_loss).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def train(
    config: DeepCFRConfig, rng: np.random.Generator, out_dir: str,
    checkpoint_interval: int = 1,
    eval_interval: int = 0,
    eval_fn: Optional[Callable[[cfr_networks.AdvantageNet, int], None]] = None,
) -> cfr_networks.AdvantageNet:
    """Runs `config.iterations` outer iterations, each `traversals_per_iteration`
    Monte Carlo traversals feeding the shared reservoir followed by
    `sgd_steps_per_iteration` minibatch gradient steps on the shared
    advantage net. Saves a checkpoint (cfr_networks.save) to
    `<out_dir>/checkpoint_latest` every `checkpoint_interval` iterations.
    `eval_fn(net, iteration)`, if given, is called every `eval_interval`
    iterations -- left as an injected callback (rather than importing
    cfr_policy/tournament here) so this module doesn't need to depend on
    the eval/inference stack to do its one job of training."""
    feature_indices = cfr_features.feature_indices(config.feature_keys)
    net = cfr_networks.AdvantageNet(input_dim=len(feature_indices), hidden_sizes=config.hidden_sizes)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    reservoir = cfr_reservoir.ReservoirBuffer(
        config.reservoir_capacity, len(feature_indices), strategy.NUM_ACTION_CATEGORIES, rng,
    )
    net_config = cfr_networks.AdvantageNetConfig(
        feature_keys=config.feature_keys, hidden_sizes=config.hidden_sizes, table_size=config.table_size,
    )

    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, "checkpoint_latest")

    for iteration in range(1, config.iterations + 1):
        t0 = time.time()
        for _ in range(config.traversals_per_iteration):
            cfr_tree.traverse_hand(
                net, reservoir, config.table_size, config.game_config, rng, float(iteration), feature_indices,
                num_equity_rollouts=config.num_equity_rollouts,
            )

        net.train()
        losses = []
        for _ in range(config.sgd_steps_per_iteration):
            if len(reservoir) == 0:
                break
            losses.append(_train_step(net, optimizer, reservoir, config.batch_size, iteration))

        elapsed = time.time() - t0
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        print(
            f"iter {iteration:4d} | reservoir {len(reservoir):7d}/{config.reservoir_capacity} | "
            f"mean loss {mean_loss:10.4f} | {elapsed:5.1f}s"
        )

        if checkpoint_interval > 0 and iteration % checkpoint_interval == 0:
            cfr_networks.save(net, net_config, checkpoint_path)

        if eval_fn is not None and eval_interval > 0 and iteration % eval_interval == 0:
            eval_fn(net, iteration)

    cfr_networks.save(net, net_config, checkpoint_path)
    return net
