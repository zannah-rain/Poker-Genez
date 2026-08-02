"""The evolvable unit: a per-street, priority-ordered list of Rule objects
(see rules.py) -- the kind of simplified strategy a human could actually
memorize and execute at a table, rather than doing live arithmetic.

This replaces an earlier array-based representation (condition_features/
condition_buckets/rule_actions/rule_mix_actions/preflop_hole_category_mask/
gto_flags numpy arrays, plus a genome-wide num_buckets/thresholds bucketing
scheme, all mutated/crossed-over via free functions in strategy.py) with
Rule/Action/Condition objects (rules.py) that ARE the genes themselves --
not a cached view over arrays. Mutation and crossover now operate directly
on these object graphs; see rules.py's module docstring for the full
rationale (per-condition bucketing instead of genome-wide, variable
condition/rule counts instead of fixed-size, an evolvable per-rule
priority).

The decision rule, in full (see rules.py for the machinery this delegates to):

  1. Every "condition" feature (the ~49 generalized concepts in features.py
     usable as a rule condition, not its ~150 one-hot indicator children,
     and not street -- see point 2) stays on its raw normalized 0-1 scale
     until a Condition buckets it with its own evolved num_buckets/
     thresholds -- each Condition owns this independently now, so two
     conditions on the same feature (in different rules) can use different
     cut points.
  2. Pick the rule pool for the current street: rules live in one of 4
     separate pools (Preflop/Flop/Turn/River) rather than one shared pool,
     so a genome can't spend its whole rule budget on preflop and leave the
     river bare. Which street a rule belongs to is therefore never a
     condition -- it's simply which pool it lives in. Each street's pool
     size is itself evolvable, independently per street per genome, within
     [rules.MIN_RULES_PER_STREET, rules.MAX_RULES_PER_STREET].
  3. Within that street's pool, check its rules in priority order (each
     StandardRule carries an evolvable `priority` float; a street's rules
     are always stored pre-sorted descending by it). Each is a conjunction
     of a variable number of (feature, required bucket) checks -- bounded
     by [rules.MIN_CONDITIONS_PER_RULE, rules.MAX_CONDITIONS_PER_RULE], also
     evolvable. Every *preflop* rule also always carries a mandatory
     hole_category_mask -- which of the 12 hole-hand-category buckets
     (Premium Pairs, Axs, Suited Connectors, Junk, etc. --
     features.py's hole_hand_category_norm) it applies to, checked exactly
     (not bucketed) since that's the one axis a real preflop range chart is
     built around. First full match wins.
  4. That rule's action -- Fold / Call / one of 6 fixed-size Raises
     (25/50/75/100/125/150% pot) / All-In -- is what gets played, *or* a
     "Mix" of two such actions, played 50/50 at decision time -- the only
     source of randomized behavior in a genome's decisions. No rule
     matching at all defaults to Fold.

Every decision re-evaluates its street's rule pool fresh from the current
Situation -- there's no memory of "I'm currently bluffing" carried between
streets. "Raise now, fold if raised back" isn't a thing this system
represents explicitly: it just falls out of a *later* decision (on the same
or a later street) landing in a different pool, or a bigger call_amount/
facing_bet naturally routing to a different (likely Fold) rule within it.
Likewise "call up to X, else fold" is just an ordinary rule condition on the
existing call_amount_norm feature's bucket, not a special parameter -- once
the bet gets too big for that condition to match, control falls through to
whatever rule the genome evolved as its fallback (typically Fold).

On top of this general strategy, a genome can also memorize exact charts for
specific, well-defined spots (see gto.py) -- e.g. "UTG open, 100BB":
situations narrow enough that a human plays them from a memorized range
chart rather than by feel. This piece keeps its historical behavior: an
active GTOSpotRule (rules.py) is always a hard override, checked first,
unconditionally, ahead of the street's StandardRule list -- GTOSpotRules
don't carry a priority and never interleave with StandardRules.
"""

from __future__ import annotations

import json

import numpy as np

import rules
import strategy
from features import Situation, extract_features

FOLD, CHECK_CALL, BET_RAISE = 0, 1, 2
ACTION_NAMES = ["fold", "check/call", "bet/raise"]

# Fraction of GTO_SPOTS a freshly random genome starts out trusting -- low,
# so the GA has to actively select for a spot's chart being useful (mirrors
# the "earn complexity, don't start with it" philosophy behind
# rules.CONDITION_GROWTH_INIT_PROB) while still giving mutation/crossover
# some initial material.
GTO_INIT_PROB = 0.1


def _apply_decision(decision: rules.Decision, legal_actions: list[int]) -> tuple[int, float]:
    """Converts a rule's Decision into a (game_action, bet_size) pair,
    applying the shared legal-action fallbacks (Fold -> Check/Call if
    folding isn't legal, Raise -> Check/Call if raising isn't legal) --
    used identically whether the Decision came from a GTOSpotRule or a
    StandardRule."""
    if decision.kind == "fold":
        action = FOLD if FOLD in legal_actions else CHECK_CALL
        return action, 0.0
    if decision.kind == "call":
        return CHECK_CALL, 0.0
    # "raise"
    if BET_RAISE not in legal_actions:
        return CHECK_CALL, 0.0
    return BET_RAISE, decision.bet_size


def _sorted_by_priority(street_rules: tuple[rules.StandardRule, ...]) -> tuple[rules.StandardRule, ...]:
    return tuple(sorted(street_rules, key=lambda r: -r.priority))


def _repair_rule_count(
    street_rules: tuple[rules.StandardRule, ...],
    rng: np.random.Generator,
    eligible_indices: np.ndarray,
    hole_category_init_prob: float | None,
) -> tuple[rules.StandardRule, ...]:
    """Clamps a street's rule tuple into [MIN_RULES_PER_STREET,
    MAX_RULES_PER_STREET]: randomly drops excess if over MAX, pads with
    fresh random rules if under MIN. Used after mutation's grow/shrink
    rolls, crossover's pooled 50% inclusion, and loading a save whose rule
    count falls outside the current bounds."""
    if len(street_rules) > rules.MAX_RULES_PER_STREET:
        keep_idx = sorted(rng.choice(len(street_rules), size=rules.MAX_RULES_PER_STREET, replace=False))
        street_rules = tuple(street_rules[i] for i in keep_idx)
    elif len(street_rules) < rules.MIN_RULES_PER_STREET:
        extra = tuple(
            rules.StandardRule.random(rng, eligible_indices, hole_category_init_prob)
            for _ in range(rules.MIN_RULES_PER_STREET - len(street_rules))
        )
        street_rules = street_rules + extra
    return street_rules


class Genome:
    """rules_by_street: tuple[tuple[rules.StandardRule, ...], ...], length
    strategy.NUM_STREETS -- each street's rules (0=preflop, 1=flop, 2=turn,
    3=river, matching Situation.street), always stored pre-sorted
    descending by priority. gto_rules: tuple[rules.GTOSpotRule, ...], one
    per gto.py GTO_SPOTS catalog entry, catalog order, always all present
    (inactive ones just have active=False)."""

    __slots__ = ("rules_by_street", "gto_rules")

    def __init__(
        self,
        rules_by_street: tuple[tuple[rules.StandardRule, ...], ...],
        gto_rules: tuple[rules.GTOSpotRule, ...],
    ):
        self.rules_by_street = rules_by_street
        self.gto_rules = gto_rules

    @classmethod
    def random(cls, rng: np.random.Generator, scale: float = 0.5) -> "Genome":
        """`scale` (relative to its default of 0.5) scales how many
        conditions a freshly random rule starts with -- there's no "weight
        magnitude" analog to scale in this representation, but this
        preserves scale's role as an initial-complexity knob. Each street
        starts at rules.MAX_RULES_PER_STREET rules (not MIN) -- mirrors
        today's proven-working scale (many rules, each individually sparse)
        rather than starting from one rule and hoping growth mutation
        discovers a useful count fast enough; the sparsity-style complexity
        penalty (nonzero_weight_count(), driving main.py's
        --sparsity-penalty) already provides pressure to shrink where a
        smaller ruleset performs just as well."""
        condition_growth_prob = float(np.clip(rules.CONDITION_GROWTH_INIT_PROB * (scale / 0.5), 0.0, 1.0))

        rules_by_street = []
        for street in range(strategy.NUM_STREETS):
            eligible = strategy.ELIGIBLE_CONDITION_INDICES_BY_STREET[street]
            hole_category_init_prob = rules.HOLE_CATEGORY_INIT_PROB if street == strategy.PREFLOP else None
            street_rules = tuple(
                rules.StandardRule.random(rng, eligible, hole_category_init_prob, condition_growth_prob)
                for _ in range(rules.MAX_RULES_PER_STREET)
            )
            rules_by_street.append(_sorted_by_priority(street_rules))

        gto_rules = rules.GTOSpotRule.catalog(rng, GTO_INIT_PROB)
        return cls(tuple(rules_by_street), gto_rules)

    def to_dict(self) -> dict:
        """Named-dictionary save form -- keyed by feature/GTO-spot/street
        name rather than array position, so a saved genome survives
        features.py/gto.py entries being added, removed, or reordered (see
        from_dict, the corresponding loader). Unlike the old array-based
        schema, there's no separate genome-wide "feature_buckets" section --
        each condition now carries its own num_buckets/thresholds inline,
        since bucketing is a per-condition gene, not a shared one."""
        rules_dict = {}
        for street, street_label in enumerate(strategy.STREET_LABELS):
            street_rules = []
            for rule in self.rules_by_street[street]:
                conditions = [
                    {
                        "feature": strategy.CONDITION_FEATURES[c.feature_index].key,
                        "num_buckets": c.num_buckets,
                        "thresholds": [float(x) for x in c.thresholds],
                        "bucket": c.bucket,
                    }
                    for c in rule.conditions
                ]
                primary_idx, mix_idx = rule.action.to_indices()
                rule_dict = {
                    "conditions": conditions,
                    "action": strategy.ACTION_CATEGORIES[primary_idx],
                    "mix_action": strategy.ACTION_CATEGORIES[mix_idx] if mix_idx != strategy.NO_MIX else None,
                    "priority": rule.priority,
                }
                if street == strategy.PREFLOP:
                    rule_dict["hole_categories"] = [
                        label for label, flag in zip(strategy.HOLE_CATEGORY_LABELS, rule.hole_category_mask) if flag
                    ]
                street_rules.append(rule_dict)
            rules_dict[street_label] = street_rules

        return {
            "rules": rules_dict,
            "gto_flags": {r.spot.key: (1.0 if r.active else 0.0) for r in self.gto_rules},
        }

    @classmethod
    def from_dict(cls, data: dict, rng: np.random.Generator) -> "Genome":
        """Reconstructs a Genome from its named-dictionary save form (see
        to_dict). Robust to the feature/GTO-spot catalog having changed
        since this genome was saved, and to the current MIN/MAX condition-
        and rule-count bounds differing from when it was saved: entries
        whose name no longer exists are dropped (with a warning); a rule
        left with zero conditions after dropping gets one fresh random
        condition (same repair a freshly random or freshly mutated
        all-wildcard rule gets); a street's rule count outside
        [MIN_RULES_PER_STREET, MAX_RULES_PER_STREET] gets clamped (with a
        warning) -- otherwise a saved genome's rule count is used exactly
        as saved, no forced fixed count. A save from before rules were
        split per street (a flat list rather than a per-street dict) can't
        be meaningfully mapped onto the new pools, so it's treated as "no
        rules saved" rather than misassigned."""
        saved_rules_by_street = data.get("rules", {})
        if not isinstance(saved_rules_by_street, dict):
            saved_rules_by_street = {}

        action_index_by_name = {name: i for i, name in enumerate(strategy.ACTION_CATEGORIES)}
        hole_category_index_by_label = {label: i for i, label in enumerate(strategy.HOLE_CATEGORY_LABELS)}

        unknown_rule_features: set[str] = set()
        ineligible_preflop_features: set[str] = set()
        unknown_hole_categories: set[str] = set()

        rules_by_street = []
        for street, street_label in enumerate(strategy.STREET_LABELS):
            eligible = strategy.ELIGIBLE_CONDITION_INDICES_BY_STREET[street]
            eligible_set = {int(x) for x in eligible}
            hole_category_init_prob = rules.HOLE_CATEGORY_INIT_PROB if street == strategy.PREFLOP else None

            saved_street_rules = saved_rules_by_street.get(street_label, [])
            if not isinstance(saved_street_rules, list):
                saved_street_rules = []

            parsed_rules = []
            for rule_data in saved_street_rules:
                conditions = []
                for cond in rule_data.get("conditions", []):
                    feature_key = cond.get("feature")
                    if feature_key is None:
                        continue
                    try:
                        fi = strategy.feature_index(feature_key)
                    except KeyError:
                        unknown_rule_features.add(feature_key)
                        continue
                    if street == strategy.PREFLOP and fi not in eligible_set:
                        ineligible_preflop_features.add(feature_key)
                        continue

                    is_boolean = strategy.CONDITION_FEATURES[fi].kind == "boolean"
                    if is_boolean:
                        num_buckets = 2
                        thresholds = (0.5,) * (strategy.MAX_BUCKETS - 1)
                    else:
                        num_buckets = int(cond.get("num_buckets", strategy.MIN_BUCKETS))
                        num_buckets = min(max(num_buckets, strategy.MIN_BUCKETS), strategy.MAX_BUCKETS)
                        saved_thresholds = [float(x) for x in cond.get("thresholds", [])]
                        padded = (saved_thresholds + [0.5] * strategy.MAX_BUCKETS)[: strategy.MAX_BUCKETS - 1]
                        thresholds = tuple(sorted(padded))
                    bucket = int(cond.get("bucket", 0))
                    conditions.append(rules.Condition(fi, num_buckets, thresholds, bucket))

                conditions = conditions[: rules.MAX_CONDITIONS_PER_RULE]
                while len(conditions) < rules.MIN_CONDITIONS_PER_RULE:
                    conditions.append(rules.Condition.random(rng, eligible))

                primary_idx = action_index_by_name.get(rule_data.get("action"), strategy.ACTION_FOLD)
                mix_name = rule_data.get("mix_action")
                mix_idx = action_index_by_name.get(mix_name, strategy.NO_MIX) if mix_name is not None else strategy.NO_MIX
                action = rules.SingleAction(primary_idx) if mix_idx == strategy.NO_MIX else rules.MixAction(primary_idx, mix_idx)

                if "priority" in rule_data:
                    priority = float(rule_data["priority"])
                else:
                    priority = float(rng.random())

                hole_category_mask = None
                if street == strategy.PREFLOP:
                    if "hole_categories" in rule_data:
                        mask = [False] * strategy.NUM_HOLE_CATEGORIES
                        for label in rule_data["hole_categories"]:
                            if label in hole_category_index_by_label:
                                mask[hole_category_index_by_label[label]] = True
                            else:
                                unknown_hole_categories.add(label)
                    else:
                        print(
                            f"Warning: a Preflop rule in saved genome has no 'hole_categories' -- "
                            "initializing randomly."
                        )
                        mask = [bool(rng.random() < rules.HOLE_CATEGORY_INIT_PROB) for _ in range(strategy.NUM_HOLE_CATEGORIES)]
                    hole_category_mask = tuple(mask)

                parsed_rules.append(rules.StandardRule(tuple(conditions), action, hole_category_mask, priority))

            original_count = len(parsed_rules)
            repaired = _repair_rule_count(tuple(parsed_rules), rng, eligible, hole_category_init_prob)
            if len(repaired) != original_count:
                print(
                    f"Warning: saved genome has {original_count} {street_label} rule(s), current bounds are "
                    f"[{rules.MIN_RULES_PER_STREET}, {rules.MAX_RULES_PER_STREET}] -- "
                    f"{'dropping excess' if original_count > len(repaired) else 'padding with fresh random rules'}."
                )
            rules_by_street.append(_sorted_by_priority(repaired))

        if unknown_rule_features:
            print(
                f"Warning: ignoring {len(unknown_rule_features)} unknown feature(s) referenced by saved "
                f"genome's rules (no longer in the feature catalog): {', '.join(sorted(unknown_rule_features))}"
            )
        if ineligible_preflop_features:
            print(
                f"Warning: ignoring {len(ineligible_preflop_features)} feature(s) referenced by saved "
                "genome's Preflop rules that aren't allowed there (need a board, or something to compare "
                f"the hole cards against, that doesn't exist preflop): {', '.join(sorted(ineligible_preflop_features))}"
            )
        if unknown_hole_categories:
            print(
                f"Warning: ignoring {len(unknown_hole_categories)} unknown hole hand category label(s) in "
                f"saved genome's preflop rules (no longer in the catalog): {', '.join(sorted(unknown_hole_categories))}"
            )

        saved_gto = data.get("gto_flags", {})
        gto_rules_tuple, unknown_gto, missing_gto = rules.gto_rules_from_saved(saved_gto, rng, GTO_INIT_PROB)
        if unknown_gto:
            print(
                f"Warning: ignoring {len(unknown_gto)} unknown GTO spot(s) in saved genome "
                f"(no longer in the catalog): {', '.join(sorted(unknown_gto))}"
            )
        if missing_gto:
            print(
                f"Warning: {len(missing_gto)} GTO spot(s) missing from saved genome (new "
                f"since this genome was saved) -- initializing randomly: {', '.join(missing_gto)}"
            )

        return cls(tuple(rules_by_street), gto_rules_tuple)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str, rng: np.random.Generator | None = None) -> "Genome":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data, rng if rng is not None else np.random.default_rng())

    def copy(self) -> "Genome":
        """Cheap: every Rule/Action/Condition is immutable, so the rule/gto
        tuples can be shared by reference -- nothing about a copy can ever
        be mutated into affecting the original."""
        return Genome(self.rules_by_street, self.gto_rules)

    def mutate(self, rng: np.random.Generator, rate: float, continuous_scale: float) -> "Genome":
        """Returns a mutated copy. Each street's existing rules are mutated
        first (see StandardRule.mutate), then independently rolled for
        structural growth (append a fresh random rule if below
        MAX_RULES_PER_STREET) and shrink (drop a random rule if above
        MIN_RULES_PER_STREET), both damped by
        rules.STRUCTURAL_MUTATION_RATE_FACTOR relative to `rate` -- a
        structural change is a bigger move than an ordinary gene nudge, so
        it should fire less often at the same base rate. gto_rules each
        independently bit-flip `active` (see GTOSpotRule.mutate)."""
        threshold_scale = continuous_scale * strategy.THRESHOLD_MUTATION_SCALE_FACTOR
        structural_rate = rate * rules.STRUCTURAL_MUTATION_RATE_FACTOR

        rules_by_street = []
        for street in range(strategy.NUM_STREETS):
            eligible = strategy.ELIGIBLE_CONDITION_INDICES_BY_STREET[street]
            mutated = [
                rule.mutate(rng, rate, eligible, threshold_scale) for rule in self.rules_by_street[street]
            ]
            if len(mutated) < rules.MAX_RULES_PER_STREET and rng.random() < structural_rate:
                hole_category_init_prob = rules.HOLE_CATEGORY_INIT_PROB if street == strategy.PREFLOP else None
                mutated.append(rules.StandardRule.random(rng, eligible, hole_category_init_prob))
            if len(mutated) > rules.MIN_RULES_PER_STREET and rng.random() < structural_rate:
                mutated.pop(int(rng.integers(0, len(mutated))))
            rules_by_street.append(_sorted_by_priority(tuple(mutated)))

        gto_rules = tuple(rule.mutate(rng, rate) for rule in self.gto_rules)

        return Genome(tuple(rules_by_street), gto_rules)

    def crossover(self, other: "Genome", rng: np.random.Generator) -> "Genome":
        """Returns a child combining self and other. Per street, pools both
        parents' rules together and independently keeps each one with 50%
        probability (rather than picking one rule per fixed slot index --
        the two parents can have different rule counts now, so there's no
        shared slot indexing to pair on); if the result falls outside
        [MIN_RULES_PER_STREET, MAX_RULES_PER_STREET] it's repaired the same
        way a saved genome's out-of-bounds count is. gto_rules: independent
        per-catalog-spot 50/50 pick of the whole object from one parent or
        the other (each spot's activation is an independent choice, so
        plain uniform crossover is the right operator, unlike the pooled
        approach above)."""
        rules_by_street = []
        for street in range(strategy.NUM_STREETS):
            eligible = strategy.ELIGIBLE_CONDITION_INDICES_BY_STREET[street]
            pool = self.rules_by_street[street] + other.rules_by_street[street]
            if pool:
                keep_mask = rng.random(len(pool)) < 0.5
                kept = tuple(rule for rule, keep in zip(pool, keep_mask) if keep)
            else:
                kept = ()
            hole_category_init_prob = rules.HOLE_CATEGORY_INIT_PROB if street == strategy.PREFLOP else None
            kept = _repair_rule_count(kept, rng, eligible, hole_category_init_prob)
            rules_by_street.append(_sorted_by_priority(kept))

        gto_mask = rng.random(len(self.gto_rules)) < 0.5
        gto_rules = tuple(
            self.gto_rules[i] if gto_mask[i] else other.gto_rules[i]
            for i in range(len(self.gto_rules))
        )

        return Genome(tuple(rules_by_street), gto_rules)

    def nonzero_weight_count(self) -> int:
        """Total condition count plus total rule count across the whole
        strategy -- how many (feature, threshold) facts, and how many
        distinct rules, a human needs to memorize to execute this genome's
        strategy. Kept under the same method name since main.py's sparsity
        penalty and tournament.py's leaderboard call it by name without
        caring about the representation; both condition count and rule
        count now feed the same penalty coefficient. Doesn't include
        gto_flags -- a different kind of complexity (how many memorized
        charts, not how many rule conditions)."""
        conditions = sum(len(rule.conditions) for street_rules in self.rules_by_street for rule in street_rules)
        rule_count = sum(len(street_rules) for street_rules in self.rules_by_street)
        return conditions + rule_count

    def active_gto_spots(self) -> list:
        """The GTO_SPOTS entries this genome currently trusts (active), in
        catalog order -- the same order decide() checks them in."""
        return [r.spot for r in self.gto_rules if r.active]

    def force_all_gto_active(self) -> None:
        """In-place: sets every GTOSpotRule to active, overriding whatever
        mutation/crossover produced. Used by ga.py's IslandModel to keep
        --force-gto-islands islands always trusting every GTO_SPOTS chart
        instead of letting gto_rules evolve freely."""
        self.gto_rules = tuple(rules.GTOSpotRule(r.spot, True) for r in self.gto_rules)

    def decide(
        self,
        situation: Situation,
        legal_actions: list[int],
        rng: np.random.Generator | None = None,
    ) -> tuple[int, float]:
        """Returns (action, raw_bet_size_in_chips_if_betting_else_0)."""
        condition_values = extract_features(situation)[strategy.CONDITION_FEATURES_FULL_INDEX]
        hole_category = strategy.hole_category_index(condition_values) if situation.street == strategy.PREFLOP else -1
        context = rules.MatchContext(situation, condition_values, hole_category)

        for rule in self.gto_rules:
            decision = rule.evaluate(context, rng)
            if decision is not None:
                return _apply_decision(decision, legal_actions)

        for rule in self.rules_by_street[situation.street]:
            decision = rule.evaluate(context, rng)
            if decision is not None:
                return _apply_decision(decision, legal_actions)

        return _apply_decision(rules.Decision("fold"), legal_actions)


def save_population(genomes: list[Genome], path: str) -> None:
    """Saves a whole generation (best-first, if the caller ranked it) as one
    JSON file of named-dictionary genomes (see Genome.to_dict), so a later
    run can reload it wholesale as its starting population even after
    features.py/gto.py have changed since the save."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump([g.to_dict() for g in genomes], f)


def load_population(path: str, rng: np.random.Generator | None = None) -> list[Genome]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rng = rng if rng is not None else np.random.default_rng()
    return [Genome.from_dict(genome_data, rng) for genome_data in data]
