"""Wraps a trained AdvantageNet as a drop-in decision-maker for the real
game engine: `.decide(situation, legal_actions, rng) -> (action, bet_size)`
-- the duck-typed signature Player.genome expects (never type-checked --
see tests/test_game.py's FixedGenome), so a trained Deep CFR strategy can be
handed straight to simulate.run_session / benchmark.py, with zero changes
to that tooling.
"""

from __future__ import annotations

import numpy as np

import cfr_actions
import cfr_features
import cfr_networks
import gto
from features import Situation


class DeepCFRPolicy:
    def __init__(
        self, net: cfr_networks.AdvantageNet, feature_keys: tuple[str, ...], deterministic: bool = False,
        gto_spots: tuple[gto.GTOSpot, ...] = (),
    ):
        self.net = net
        self.feature_keys = tuple(feature_keys)
        self.feature_indices = cfr_features.feature_indices(self.feature_keys)
        # False (default): sample from the regret-matching strategy, same as
        # during training -- poker needs genuinely mixed strategies. True:
        # always take the highest-probability action, useful for a quick
        # deterministic eval/inspection run.
        self.deterministic = deterministic
        # See gto.py's module docstring -- matched decisions are played
        # exactly as fixed, same as during training (cfr_tree.py), rather
        # than asking the net. Empty by default: no override, same as before.
        self.gto_spots = gto_spots

    @classmethod
    def from_checkpoint(
        cls, path: str, deterministic: bool = False, gto_spots: tuple[gto.GTOSpot, ...] = (),
    ) -> "DeepCFRPolicy":
        net, config = cfr_networks.load(path)
        return cls(net, config.feature_keys, deterministic=deterministic, gto_spots=gto_spots)

    def decide(
        self, situation: Situation, legal_actions: list[int], rng: np.random.Generator | None = None,
    ) -> tuple[int, float]:
        fixed_action = gto.first_matching_action(self.gto_spots, situation) if self.gto_spots else None
        if fixed_action is not None:
            return cfr_actions.category_to_game_action(fixed_action, situation, legal_actions)

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
