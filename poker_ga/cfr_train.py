"""Single Deep CFR training building blocks: alternates Monte Carlo tree
traversals (cfr_tree.py, filling the shared reservoir) with minibatch
regression steps on the shared advantage network, weighting each sample's
loss by its iteration number t (see _train_step) so later iterations
dominate the training signal -- that's what gives "Single" Deep CFR its
average-strategy approximation without a second policy network/reservoir:
the *final* net's regret-matching strategy already approximates the
average strategy, the same way Linear CFR's running average strategy is
dominated by its later iterations.

Split into two layers, mirroring how main.py's CLI owns the outer GA loop
(reload, per-generation checkpointing, benchmark-vs-checkpoint, early
stopping) while delegating individual pieces to ga.py/simulate.py/
benchmark.py:

  - `Trainer` + `run_iteration`: the low-level, one-iteration-at-a-time
    building blocks. `Trainer` bundles the net/optimizer/reservoir a run
    advances (the CFR analog of ga.py's IslandModel); `run_iteration` runs
    exactly one outer iteration's traversals + gradient steps.
  - `train`: a simple convenience loop built on top of those two, for
    straightforward scripted/test use. cfr_main.py's CLI does *not* use
    this -- it drives `Trainer`/`run_iteration` directly so it can
    interleave reload-from-checkpoint and benchmark-vs-checkpoint/
    early-stopping between iterations, the same way main.py's own training
    loop does for the GA.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
from tqdm import tqdm

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


@dataclass
class Trainer:
    """Everything one outer training iteration needs to advance: the shared
    advantage net, its optimizer, and its reservoir buffer -- built once
    from a DeepCFRConfig (optionally starting from a reloaded net instead
    of a fresh random one) and then threaded through repeated
    run_iteration calls."""

    config: DeepCFRConfig
    net: cfr_networks.AdvantageNet
    optimizer: torch.optim.Optimizer
    reservoir: cfr_reservoir.ReservoirBuffer
    feature_indices: np.ndarray

    @classmethod
    def new(
        cls, config: DeepCFRConfig, rng: np.random.Generator,
        initial_net: Optional[cfr_networks.AdvantageNet] = None,
        initial_reservoir: Optional[cfr_reservoir.ReservoirBuffer] = None,
    ) -> "Trainer":
        feature_indices = cfr_features.feature_indices(config.feature_keys)
        net = initial_net if initial_net is not None else cfr_networks.AdvantageNet(
            input_dim=len(feature_indices), hidden_sizes=config.hidden_sizes,
        )
        optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
        reservoir = initial_reservoir if initial_reservoir is not None else cfr_reservoir.ReservoirBuffer(
            config.reservoir_capacity, len(feature_indices), strategy.NUM_ACTION_CATEGORIES, rng,
        )
        return cls(config=config, net=net, optimizer=optimizer, reservoir=reservoir, feature_indices=feature_indices)

    def net_config(self) -> cfr_networks.AdvantageNetConfig:
        return cfr_networks.AdvantageNetConfig(
            feature_keys=self.config.feature_keys, hidden_sizes=self.config.hidden_sizes,
            table_size=self.config.table_size,
        )

    def save(self, path: str) -> None:
        """Saves the net (cfr_networks.save, `<path>.pt`/`.json`) and its
        reservoir (cfr_reservoir.ReservoirBuffer.save, `<path>.npz`)
        together as one call -- the two should never drift out of sync, so
        there's deliberately no way to save one without the other."""
        cfr_networks.save(self.net, self.net_config(), path)
        self.reservoir.save(path)

    def revert_to(self, checkpoint_net: cfr_networks.AdvantageNet) -> None:
        """Discards the current net weights and optimizer state, replacing
        them with `checkpoint_net`'s -- used when a benchmark check (see
        cfr_main.py) shows recent training hasn't actually improved play.
        The reservoir is deliberately left untouched: the regret samples in
        it are still valid observations regardless of whether the net later
        *fit* to them turned out to help, the same way ga.py's checkpoint
        revert doesn't try to undo already-recorded fitness history, only
        which population breeds onward from here. The optimizer is rebuilt
        from scratch (not just the net's weights) since Adam's momentum/
        variance estimates were built up chasing the trajectory that's now
        being discarded -- keeping them would just push back in roughly the
        same (apparently unhelpful) direction."""
        self.net.load_state_dict(checkpoint_net.state_dict())
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr)


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


def run_iteration(trainer: Trainer, rng: np.random.Generator, iteration: int, show_progress: bool = True) -> float:
    """Runs exactly one outer iteration -- config.traversals_per_iteration
    Monte Carlo traversals feeding trainer.reservoir, then
    config.sgd_steps_per_iteration minibatch gradient steps on trainer.net
    -- and returns the mean training loss (nan if the reservoir stayed
    empty throughout, only possible on iteration 1 with an unusually small
    traversals_per_iteration). These two loops are the slowest part of a
    training run (a full recursive tree traversal per hand; a forward+
    backward pass per gradient step), so both get their own tqdm bar --
    `leave=False` so a long run's log isn't cluttered with one finished bar
    per iteration."""
    config = trainer.config
    for _ in tqdm(
        range(config.traversals_per_iteration), desc=f"iter {iteration} traversals",
        unit="hand", disable=not show_progress, leave=False,
    ):
        cfr_tree.traverse_hand(
            trainer.net, trainer.reservoir, config.table_size, config.game_config, rng,
            float(iteration), trainer.feature_indices, num_equity_rollouts=config.num_equity_rollouts,
        )

    trainer.net.train()
    losses = []
    for _ in tqdm(
        range(config.sgd_steps_per_iteration), desc=f"iter {iteration} sgd steps",
        unit="step", disable=not show_progress, leave=False,
    ):
        if len(trainer.reservoir) == 0:
            break
        losses.append(_train_step(trainer.net, trainer.optimizer, trainer.reservoir, config.batch_size, iteration))
    return float(np.mean(losses)) if losses else float("nan")


def train(
    config: DeepCFRConfig, rng: np.random.Generator, out_dir: str,
    checkpoint_interval: int = 1,
    eval_interval: int = 0,
    eval_fn: Optional[Callable[[cfr_networks.AdvantageNet, int], None]] = None,
    initial_net: Optional[cfr_networks.AdvantageNet] = None,
    initial_reservoir: Optional[cfr_reservoir.ReservoirBuffer] = None,
    show_progress: bool = True,
) -> cfr_networks.AdvantageNet:
    """Simple convenience loop for straightforward scripted/test use: runs
    config.iterations calls to run_iteration back to back, checkpointing
    (Trainer.save -- net and reservoir together) to `<out_dir>/checkpoint_latest`
    every `checkpoint_interval` iterations and calling `eval_fn(net, iteration)`
    (if given) every `eval_interval` iterations -- left as an injected
    callback (rather than importing cfr_policy/tournament here) so this
    module doesn't need to depend on the eval/inference stack to do its one
    job of training. `initial_net`/`initial_reservoir`, if given, resume
    training from an already-trained net/reservoir instead of fresh ones.

    cfr_main.py's CLI does *not* use this function -- see the module
    docstring for why it drives Trainer/run_iteration directly instead."""
    trainer = Trainer.new(config, rng, initial_net=initial_net, initial_reservoir=initial_reservoir)
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, "checkpoint_latest")

    for iteration in range(1, config.iterations + 1):
        t0 = time.time()
        mean_loss = run_iteration(trainer, rng, iteration, show_progress=show_progress)
        elapsed = time.time() - t0
        print(
            f"iter {iteration:4d} | reservoir {len(trainer.reservoir):7d}/{config.reservoir_capacity} | "
            f"mean loss {mean_loss:10.4f} | {elapsed:5.1f}s"
        )

        if checkpoint_interval > 0 and iteration % checkpoint_interval == 0:
            trainer.save(checkpoint_path)

        if eval_fn is not None and eval_interval > 0 and iteration % eval_interval == 0:
            eval_fn(trainer.net, iteration)

    trainer.save(checkpoint_path)
    return trainer.net
