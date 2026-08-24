"""Glue between Single Deep CFR's 9-category discrete action set
(strategy.ACTION_CATEGORIES) and the game engine's 3-way legal_actions
(FOLD/CHECK_CALL/BET_RAISE, genome.py) -- the one place this touches game
mechanics, shared by both training (cfr_tree.py) and inference
(cfr_policy.py) so they can never drift apart.

Reuses rules._standard_decision and genome._apply_decision as-is: those
already implement exactly "ACTION_CATEGORIES index -> Decision" and
"Decision -> (game_action, bet_size), with illegal-action fallback"
respectively.
"""

from __future__ import annotations

import numpy as np

import strategy
from features import Situation
from genome import BET_RAISE, CHECK_CALL, FOLD, _apply_decision
from rules import Decision, _standard_decision

NUM_ACTIONS = strategy.NUM_ACTION_CATEGORIES


def legal_action_categories(legal_actions: list[int]) -> np.ndarray:
    """Boolean mask, shape (NUM_ACTIONS,), over strategy.ACTION_CATEGORIES:
    which discrete categories are meaningfully choosable given the game
    engine's legal_actions at this decision. Mirrors _apply_decision's own
    fallback rules (Fold -> Check/Call if illegal, Raise -> Check/Call if
    illegal) by construction: Call is always legal (CHECK_CALL is always in
    legal_actions -- checking for free is always an option), Fold is only
    its own category when FOLD is actually legal (otherwise it would just
    collapse to Call anyway), and every Raise size / All-In needs BET_RAISE
    to be legal."""
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[strategy.ACTION_CALL] = True
    if FOLD in legal_actions:
        mask[strategy.ACTION_FOLD] = True
    if BET_RAISE in legal_actions:
        mask[list(strategy.RAISE_ACTIONS)] = True
        mask[strategy.ACTION_ALLIN] = True
    return mask


def decision_to_game_action(decision: Decision, legal_actions: list[int]) -> tuple[int, float]:
    """A rules.Decision -> (game_action, bet_size), applying the shared
    legal-action fallbacks (Fold -> Check/Call if folding isn't legal,
    Raise -> Check/Call if raising isn't legal) -- category_to_game_action's
    own underlying step, exposed separately so a Decision built some other
    way (e.g. gto.py's fixed spots, whose raise sizes aren't always one of
    strategy.ACTION_CATEGORIES' pot-fraction buckets -- see gto.GTOSpot's
    "raise_<N>bb" tokens) resolves through the exact same fallback logic."""
    return _apply_decision(decision, legal_actions)


def category_to_game_action(
    action_index: int, situation: Situation, legal_actions: list[int]
) -> tuple[int, float]:
    """ACTION_CATEGORIES index -> (game_action, bet_size), exactly as a
    StandardRule's chosen action would resolve for the same Situation."""
    decision = _standard_decision(action_index, situation)
    return decision_to_game_action(decision, legal_actions)


def regret_matching(regrets: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    """Standard regret matching: probability of each legal action
    proportional to its positive regret, zero on illegal actions. Falls back
    to uniform over legal actions when no legal action has positive regret
    (e.g. at the very start of training, or a node with all-negative
    regrets)."""
    positive = np.where(legal_mask, np.maximum(regrets, 0.0), 0.0)
    total = positive.sum()
    if total > 1e-9:
        return positive / total
    num_legal = legal_mask.sum()
    return legal_mask.astype(np.float64) / num_legal


def regret_matching_batch(regrets: np.ndarray, legal_masks: np.ndarray) -> np.ndarray:
    """Vectorized regret_matching over a whole batch at once -- same
    positive-regret-normalized, else-uniform-over-legal logic, row by row,
    but without a Python-level loop. `regrets`/`legal_masks`: shape
    (n, NUM_ACTIONS). Returns shape (n, NUM_ACTIONS). Used where
    regret_matching would otherwise need calling once per row over
    thousands+ of samples (e.g. cfr_explorer.py computing the current net's
    action probabilities for a whole loaded reservoir up front)."""
    positive = np.where(legal_masks, np.maximum(regrets, 0.0), 0.0)
    totals = positive.sum(axis=1, keepdims=True)
    num_legal = legal_masks.sum(axis=1, keepdims=True)
    uniform = legal_masks.astype(np.float64) / num_legal
    has_positive = totals > 1e-9
    return np.where(has_positive, positive / np.where(has_positive, totals, 1.0), uniform)
