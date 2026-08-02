"""Shared test-support builders for constructing StandardRule/Condition/
Genome fixtures without hand-rolling the same object-graph boilerplate in
every test file. Every StandardRule now needs at least one real condition
(no wildcard sentinel exists anymore -- see rules.py), so these helpers
lean on `tautological_condition` (a condition engineered to match any
legal value) to isolate whatever a test is actually exercising (which
action fires, priority ordering, hole-category gating, GTO overrides, ...)
from ordinary condition matching, the same role the old array-based
"wildcard" conditions played.
"""

from __future__ import annotations

import rules
import strategy
from genome import Genome
from gto import GTO_SPOTS


def tautological_condition(feature_index: int) -> rules.Condition:
    """A Condition that matches literally any legal (0-1 normalized) value
    for the given feature: num_buckets=2 with a threshold at 0.0 means
    every value lands in bucket 1 (compute_bucket's searchsorted(...,
    side="right") counts a value exactly equal to a cut as "at or above"
    it, and features.py never normalizes below 0.0)."""
    return rules.Condition(feature_index, 2, (0.0, 0.0), 1)


def condition(feature_key: str, bucket: int, num_buckets: int = 2, thresholds: tuple = (0.5, 0.5)) -> rules.Condition:
    """A real condition on a named feature -- defaults (num_buckets=2,
    threshold at 0.5) match the convention used for boolean features
    (bucket 1 = true, bucket 0 = false)."""
    return rules.Condition(strategy.feature_index(feature_key), num_buckets, thresholds, bucket)


def make_rule(
    action_index: int = strategy.ACTION_FOLD,
    mix_index: int = strategy.NO_MIX,
    conditions: list[rules.Condition] | None = None,
    priority: float = 0.5,
    hole_categories: list[str] | None = None,
    street: int = strategy.PREFLOP,
) -> rules.StandardRule:
    """A StandardRule. Defaults to a single tautological condition (see
    above) so it isolates whatever the test is actually exercising; pass
    `conditions` (a list of real Condition objects, e.g. via `condition()`)
    to test actual matching behavior instead.

    hole_categories: for preflop rules (street == strategy.PREFLOP), a list
    of HOLE_CATEGORY_LABELS this rule claims; None means "claims every
    category" (behaves like a wildcard on that axis unless a test overrides
    it). Ignored for non-preflop rules (hole_category_mask stays None)."""
    if conditions is None:
        eligible = strategy.ELIGIBLE_CONDITION_INDICES_BY_STREET[street]
        conditions = [tautological_condition(int(eligible[0]))]

    action = rules.SingleAction(action_index) if mix_index == strategy.NO_MIX else rules.MixAction(action_index, mix_index)

    hole_category_mask = None
    if street == strategy.PREFLOP:
        if hole_categories is None:
            hole_category_mask = tuple(True for _ in strategy.HOLE_CATEGORY_LABELS)
        else:
            claimed = set(hole_categories)
            hole_category_mask = tuple(label in claimed for label in strategy.HOLE_CATEGORY_LABELS)

    return rules.StandardRule(tuple(conditions), action, hole_category_mask, priority)


def _sorted_by_priority(street_rules: tuple[rules.StandardRule, ...]) -> tuple[rules.StandardRule, ...]:
    return tuple(sorted(street_rules, key=lambda r: -r.priority))


def make_genome(
    rules_by_street: dict[int, tuple[rules.StandardRule, ...]] | None = None,
    gto_active: dict[str, bool] | None = None,
    gto_rules: tuple[rules.GTOSpotRule, ...] | None = None,
) -> Genome:
    """Defaults to an "empty" genome: every street has exactly one rule (an
    always-matching, always-Fold rule -- see make_rule), no GTO charts
    trusted. `rules_by_street`: {street_index: (rule, ...)} overrides for
    specific streets (each street's tuple is re-sorted by priority, mirroring
    what Genome.random()/mutate()/crossover() always do -- decide() relies
    on stored order, it doesn't sort at decision time); unlisted streets
    keep the default single Fold rule. `gto_active`: {spot_key: bool}
    overrides against the real GTO_SPOTS catalog; unlisted spots stay
    inactive. `gto_rules`: bypasses the GTO_SPOTS catalog entirely with an
    explicit tuple -- for tests exercising a synthetic spot not in the real
    catalog (mutually exclusive with gto_active)."""
    rules_by_street = rules_by_street or {}
    street_tuples = []
    for street in range(strategy.NUM_STREETS):
        if street in rules_by_street:
            street_tuples.append(_sorted_by_priority(tuple(rules_by_street[street])))
        else:
            street_tuples.append((make_rule(strategy.ACTION_FOLD, street=street),))

    if gto_rules is not None:
        gto = tuple(gto_rules)
    else:
        gto_active = gto_active or {}
        gto = tuple(rules.GTOSpotRule(spot, gto_active.get(spot.key, False)) for spot in GTO_SPOTS)

    return Genome(tuple(street_tuples), gto)


def genome_with_default_action(action: int, **kwargs) -> Genome:
    """A genome whose single rule on every street matches every situation
    and always plays `action` -- isolates decide()'s action-category ->
    (game action, bet size) mapping from condition-matching/street-routing/
    priority machinery, which is tested separately."""
    rules_by_street = {
        street: (make_rule(action, street=street),) for street in range(strategy.NUM_STREETS)
    }
    return make_genome(rules_by_street=rules_by_street, **kwargs)


def genome_with_condition(feature_key: str, bucket: int, action: int, street: int = strategy.PREFLOP, **kwargs) -> Genome:
    """A genome whose single rule on `street` (preflop by default) requires
    exactly one (feature, bucket) match to play `action`; every other
    street keeps the default single always-matching Fold rule, and a
    non-matching situation on the tested street falls through to
    decide()'s own no-match-defaults-to-Fold behavior (there's no fallback
    rule needed -- Genome.decide() already defaults to Fold when a street's
    whole rule list comes up empty)."""
    rule = make_rule(action, conditions=[condition(feature_key, bucket)], street=street)
    return make_genome(rules_by_street={street: (rule,)}, **kwargs)
