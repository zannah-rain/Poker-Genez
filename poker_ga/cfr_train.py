"""Single Deep CFR training building blocks: alternates Monte Carlo tree
traversals (cfr_tree.py, filling the shared reservoir) with minibatch
regression steps on the shared advantage network, weighting each sample's
loss by its iteration number t (see _train_step) so later iterations
dominate the training signal -- that's what gives "Single" Deep CFR its
average-strategy approximation without a second policy network/reservoir:
the *final* net's regret-matching strategy already approximates the
average strategy, the same way Linear CFR's running average strategy is
dominated by its later iterations.

Split into two layers:

  - `Trainer` + `run_iteration`: the low-level, one-iteration-at-a-time
    building blocks. `Trainer` bundles the net/optimizer/reservoir a run
    advances; `run_iteration` runs exactly one outer iteration's
    traversals + gradient steps.
  - `train`: a simple convenience loop built on top of those two, for
    straightforward scripted/test use. cfr_main.py's CLI does *not* use
    this -- it drives `Trainer`/`run_iteration` directly so it can
    interleave reload-from-checkpoint and benchmark-vs-checkpoint/
    early-stopping between iterations.
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
    dropout: float = cfr_networks.DEFAULT_DROPOUT
    weight_decay: float = 1e-5
    # torch.nn.utils.clip_grad_norm_'s max_norm, applied every _train_step
    # before optimizer.step() -- 0 disables clipping. Guards against exactly
    # the failure mode _train_step's own comment describes: Adam's second-
    # moment estimate lagging a gradient scale that's climbing (e.g. from a
    # bad minibatch, or a distribution shift as the reservoir fills), which
    # otherwise shows up as a sudden, hard-to-recover loss spike.
    grad_clip_norm: float = 5.0
    reservoir_capacity: int = 200_000
    num_equity_rollouts: int = cfr_tree.DEFAULT_NUM_EQUITY_ROLLOUTS
    min_starting_stack_bb: float = cfr_tree.DEFAULT_MIN_STARTING_STACK_BB
    max_starting_stack_bb: float = cfr_tree.DEFAULT_MAX_STARTING_STACK_BB
    game_config: GameConfig = field(default_factory=GameConfig)


def _trainer_state_path(path: str) -> str:
    return f"{path}_trainer_state.pt"


@dataclass
class Trainer:
    """Everything one outer training iteration needs to advance: the shared
    advantage net, its optimizer, and its reservoir buffer -- built once
    from a DeepCFRConfig (optionally starting from a reloaded net instead
    of a fresh random one) and then threaded through repeated
    run_iteration calls.

    completed_iterations is how many outer iterations this net/optimizer
    have actually been trained through -- run_iteration keeps it in sync
    (see its own docstring). It's not just bookkeeping: every reservoir
    sample is stored with the outer iteration `t` it was collected at (see
    cfr_reservoir.ReservoirBuffer.add) and _train_step normalizes those by
    the *current* outer iteration to get Linear CFR's recency weighting
    (see _train_step's own comment). A resumed run whose iteration counter
    restarted at 1 while its reloaded reservoir's `t` values were still on
    the old run's scale (e.g. up to a few hundred) would divide by a much
    smaller number than the samples were collected under -- inflating old
    samples' weights by up to that old-scale/1 ratio and spiking the
    reported loss -- so a resume must carry this forward and keep counting
    from it, not restart at 1."""

    config: DeepCFRConfig
    net: cfr_networks.AdvantageNet
    optimizer: torch.optim.Optimizer
    reservoir: cfr_reservoir.ReservoirBuffer
    feature_indices: np.ndarray
    completed_iterations: int = 0

    @classmethod
    def new(
        cls, config: DeepCFRConfig, rng: np.random.Generator,
        initial_net: Optional[cfr_networks.AdvantageNet] = None,
        initial_reservoir: Optional[cfr_reservoir.ReservoirBuffer] = None,
        initial_optimizer_state: Optional[dict] = None,
        completed_iterations: int = 0,
    ) -> "Trainer":
        feature_indices = cfr_features.feature_indices(config.feature_keys)
        net = initial_net if initial_net is not None else cfr_networks.AdvantageNet(
            input_dim=len(feature_indices), hidden_sizes=config.hidden_sizes, dropout=config.dropout,
        )
        optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        if initial_optimizer_state is not None:
            optimizer.load_state_dict(initial_optimizer_state)
        reservoir = initial_reservoir if initial_reservoir is not None else cfr_reservoir.ReservoirBuffer(
            config.reservoir_capacity, len(feature_indices), strategy.NUM_ACTION_CATEGORIES, rng,
        )
        return cls(
            config=config, net=net, optimizer=optimizer, reservoir=reservoir, feature_indices=feature_indices,
            completed_iterations=completed_iterations,
        )

    def net_config(self) -> cfr_networks.AdvantageNetConfig:
        return cfr_networks.AdvantageNetConfig(
            feature_keys=self.config.feature_keys, hidden_sizes=self.config.hidden_sizes,
            table_size=self.config.table_size, dropout=self.config.dropout,
        )

    def save(self, path: str) -> None:
        """Saves the net (cfr_networks.save, `<path>.pt`/`.json`), its
        reservoir (cfr_reservoir.ReservoirBuffer.save, `<path>.npz`), and
        its optimizer state + completed_iterations (`<path>_trainer_state.pt`)
        together as one call -- the three should never drift out of sync
        (see load_trainer_state / completed_iterations' own docstring for
        why leaving either of the latter two behind corrupts a resumed
        run's loss weighting), so there's deliberately no way to save a
        subset."""
        cfr_networks.save(self.net, self.net_config(), path)
        self.reservoir.save(path)
        torch.save(
            {"optimizer": self.optimizer.state_dict(), "completed_iterations": self.completed_iterations},
            _trainer_state_path(path),
        )

    @staticmethod
    def load_trainer_state(path: str) -> tuple[Optional[dict], int]:
        """(optimizer_state_dict, completed_iterations) saved alongside
        `path` by Trainer.save, or (None, 0) if `<path>_trainer_state.pt`
        doesn't exist -- e.g. a net checkpoint saved before this was added,
        which is allowed to come back to a fresh optimizer/t=0 rather than
        treated as an error, the same way a missing reservoir file is."""
        state_path = _trainer_state_path(path)
        if not os.path.exists(state_path):
            return None, 0
        state = torch.load(state_path, map_location="cpu")
        return state["optimizer"], int(state["completed_iterations"])

    def revert_to(self, checkpoint_net: cfr_networks.AdvantageNet) -> None:
        """Discards the current net weights and optimizer state, replacing
        them with `checkpoint_net`'s -- used when a benchmark check (see
        cfr_main.py) shows recent training hasn't actually improved play.
        The reservoir is deliberately left untouched: the regret samples in
        it are still valid observations regardless of whether the net later
        *fit* to them turned out to help. The optimizer is rebuilt
        from scratch (not just the net's weights) since Adam's momentum/
        variance estimates were built up chasing the trajectory that's now
        being discarded -- keeping them would just push back in roughly the
        same (apparently unhelpful) direction."""
        self.net.load_state_dict(checkpoint_net.state_dict())
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)


def _train_step(
    net: cfr_networks.AdvantageNet, optimizer: torch.optim.Optimizer,
    reservoir: cfr_reservoir.ReservoirBuffer, batch_size: int, current_iteration: int,
    grad_clip_norm: float = 0.0,
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
    if grad_clip_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip_norm)
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
    per iteration.

    `iteration` should keep counting up from wherever `trainer` left off
    (trainer.completed_iterations before this call) rather than restart at
    1 after a resume -- see Trainer's own docstring for why a discontinuity
    there corrupts the reservoir's recency weighting. This sets
    trainer.completed_iterations = iteration at the end, so a caller that
    resumed from a reload and keeps calling this in sequence doesn't have
    to track it separately."""
    config = trainer.config
    for _ in tqdm(
        range(config.traversals_per_iteration), desc=f"iter {iteration} traversals",
        unit="hand", disable=not show_progress, leave=False,
    ):
        cfr_tree.traverse_hand(
            trainer.net, trainer.reservoir, config.table_size, config.game_config, rng,
            float(iteration), trainer.feature_indices, num_equity_rollouts=config.num_equity_rollouts,
            min_starting_stack_bb=config.min_starting_stack_bb,
            max_starting_stack_bb=config.max_starting_stack_bb,
        )

    trainer.net.train()
    losses = []
    for _ in tqdm(
        range(config.sgd_steps_per_iteration), desc=f"iter {iteration} sgd steps",
        unit="step", disable=not show_progress, leave=False,
    ):
        if len(trainer.reservoir) == 0:
            break
        losses.append(_train_step(
            trainer.net, trainer.optimizer, trainer.reservoir, config.batch_size, iteration,
            grad_clip_norm=config.grad_clip_norm,
        ))
    trainer.completed_iterations = iteration
    return float(np.mean(losses)) if losses else float("nan")


def train(
    config: DeepCFRConfig, rng: np.random.Generator, out_dir: str,
    checkpoint_interval: int = 1,
    eval_interval: int = 0,
    eval_fn: Optional[Callable[[cfr_networks.AdvantageNet, int], None]] = None,
    initial_net: Optional[cfr_networks.AdvantageNet] = None,
    initial_reservoir: Optional[cfr_reservoir.ReservoirBuffer] = None,
    initial_optimizer_state: Optional[dict] = None,
    completed_iterations: int = 0,
    show_progress: bool = True,
) -> cfr_networks.AdvantageNet:
    """Simple convenience loop for straightforward scripted/test use: runs
    config.iterations more calls to run_iteration back to back (continuing
    the outer iteration count from `completed_iterations`, not restarting
    at 1 -- see Trainer's docstring for why that matters whenever a reload
    is involved), checkpointing (Trainer.save -- net, reservoir, optimizer
    state and completed_iterations together) to
    `<out_dir>/checkpoint_latest` every `checkpoint_interval` iterations
    and calling `eval_fn(net, iteration)` (if given) every `eval_interval`
    iterations -- left as an injected callback (rather than importing
    cfr_policy/tournament here) so this module doesn't need to depend on
    the eval/inference stack to do its one job of training.
    `initial_net`/`initial_reservoir`/`initial_optimizer_state`/
    `completed_iterations`, if given, resume training from an
    already-trained net/reservoir/optimizer instead of fresh ones --
    typically all four sourced from one prior Trainer.save via
    Trainer.load_trainer_state plus cfr_networks.load/
    cfr_reservoir.ReservoirBuffer.load.

    cfr_main.py's CLI does *not* use this function -- see the module
    docstring for why it drives Trainer/run_iteration directly instead."""
    trainer = Trainer.new(
        config, rng, initial_net=initial_net, initial_reservoir=initial_reservoir,
        initial_optimizer_state=initial_optimizer_state, completed_iterations=completed_iterations,
    )
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, "checkpoint_latest")

    start = trainer.completed_iterations
    for iteration in range(start + 1, start + config.iterations + 1):
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
