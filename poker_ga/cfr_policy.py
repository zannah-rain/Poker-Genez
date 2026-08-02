"""Wraps a trained AdvantageNet as a drop-in decision-maker for the real
game engine: `.decide(situation, legal_actions, rng) -> (action, bet_size)`,
the exact same duck-typed signature genome.Genome.decide() has (Player.genome
isn't type-checked anywhere -- see tests/test_game.py's FixedGenome), so a
trained Deep CFR strategy can be handed straight to
tournament.run_final_tournament / simulate.run_session alongside GA
Players, with zero changes to that tooling.
"""

from __future__ import annotations

import numpy as np

import cfr_actions
import cfr_features
import cfr_networks
from features import Situation


class DeepCFRPolicy:
    def __init__(self, net: cfr_networks.AdvantageNet, feature_keys: tuple[str, ...], deterministic: bool = False):
        self.net = net
        self.feature_keys = tuple(feature_keys)
        self.feature_indices = cfr_features.feature_indices(self.feature_keys)
        # False (default): sample from the regret-matching strategy, same as
        # during training -- poker needs genuinely mixed strategies. True:
        # always take the highest-probability action, useful for a quick
        # deterministic eval/inspection run.
        self.deterministic = deterministic

    @classmethod
    def from_checkpoint(cls, path: str, deterministic: bool = False) -> "DeepCFRPolicy":
        net, config = cfr_networks.load(path)
        return cls(net, config.feature_keys, deterministic=deterministic)

    def decide(
        self, situation: Situation, legal_actions: list[int], rng: np.random.Generator | None = None,
    ) -> tuple[int, float]:
        legal_mask = cfr_actions.legal_action_categories(legal_actions)
        feats = cfr_features.extract_subset(situation, self.feature_indices)
        regrets = self.net.predict(feats)
        sigma = cfr_actions.regret_matching(regrets, legal_mask)
        legal_idx = np.flatnonzero(legal_mask)

        if self.deterministic:
            chosen = int(legal_idx[np.argmax(sigma[legal_idx])])
        else:
            if rng is None:
                rng = np.random.default_rng()
            p = sigma[legal_idx]
            p = p / p.sum()  # guard against float roundoff (see cfr_tree.py's identical guard)
            chosen = int(rng.choice(legal_idx, p=p))

        return cfr_actions.category_to_game_action(chosen, situation, legal_actions)
