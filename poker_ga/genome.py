"""The evolvable unit: a small, ordered list of "if this situation, then
this action" rules -- the kind of simplified strategy a human could actually
memorize and execute at a table, rather than doing live arithmetic.

This replaces an earlier linear "V/L/theta" system (two 0-100 scores per
decision, combined through evolvable thresholds) that was accurate but not
executable by a human: reading it required tracking which of ~200 weighted
features were active and summing them live, every decision, every hand.

The new decision rule, in full (see strategy.py for the machinery this
delegates to):

  1. Bucket every "condition" feature (the ~49 generalized concepts in
     features.py usable as a rule condition, not its ~150 one-hot indicator
     children, and not street -- see point 2) into 2-3 groups via this
     genome's own evolved thresholds -- shared across every rule, so e.g.
     "Hand Strength: weak/medium/strong" has one definition the whole
     strategy reads from, the way a real chart would.
  2. Pick the rule pool for the current street: rules live in one of 4
     separate, fixed-size pools (strategy.RULES_PER_STREET each) --
     Preflop/Flop/Turn/River -- rather than one shared pool, so a genome
     can't spend its whole rule budget on preflop and leave the river bare.
     Which street a rule belongs to is therefore never an optional
     condition competing for one of a rule's CONDITIONS_PER_RULE slots (or
     counting against nonzero_weight_count()'s complexity accounting) --
     it's simply which pool it lives in.
  3. Within that street's pool, check its rules in fixed array order (which
     also doubles as the priority order shown in exported reports -- see
     tournament.py); each is a small conjunction of up to CONDITIONS_PER_RULE
     (feature, required bucket) checks (or a wildcard "don't care"). Every
     *preflop* rule also always carries a mandatory hole_category_mask --
     which of the 12 hole-hand-category buckets (Premium Pairs, Axs, Suited
     Connectors, Junk, etc. -- features.py's hole_hand_category_norm) it
     applies to, checked exactly (not bucketed) since that's the one axis a
     real preflop range chart is built around. Like street, this doesn't
     count against nonzero_weight_count() either. First full match wins --
     the same idiom gto.py's GTOSpot.action_ranges/SpotMatcher already use.
  4. That rule's action -- Fold / Call / one of 6 fixed-size Raises
     (25/50/75/100/125/150% pot) / All-In -- is what gets played, *or* a
     "Mix" of two such actions, played 50/50 at decision time (see
     strategy.resolve_rule_action) -- the only source of randomized
     behavior in a genome's decisions; there's no separate noise mechanism.
     No rule matching at all defaults to Fold (mirrors GTOSpot.default_
     action's "everything not colored in is a fold").

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
chart rather than by feel. This piece is unchanged from before: `gto_flags`
is one evolvable boolean gene per catalog entry in gto.py's GTO_SPOTS ("does
this genome trust this spot's chart or not"); when a situation matches an
active spot, `decide()` looks the hand up in that spot's chart and plays
exactly what it says, bypassing the rule list entirely for that decision --
a hard override, not another rule, because a chart lookup is exactly as
memorizable for a human as the chart itself.
"""

from __future__ import annotations

import json

import numpy as np

import strategy
from features import Situation, extract_features
from gto import GTO_SPOTS, NUM_GTO_SPOTS, resolve_spot_action

FOLD, CHECK_CALL, BET_RAISE = 0, 1, 2
ACTION_NAMES = ["fold", "check/call", "bet/raise"]

# Fraction of GTO_SPOTS a freshly random genome starts out trusting -- low,
# so the GA has to actively select for a spot's chart being useful (mirrors
# the "earn complexity, don't start with it" philosophy behind
# CONDITION_ACTIVE_INIT_PROB in strategy.py) while still giving mutation/
# crossover some initial material.
GTO_INIT_PROB = 0.1


def mutate_bool_flags(flags: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Bit-flip mutation for boolean genes (0.0/1.0 floats): each selected
    gene flips to its opposite, rather than being nudged like a continuous
    value or alphabet-jumped like a categorical one -- there's nothing "in
    between" on or off. Used for preflop_hole_category_mask and gto_flags."""
    mask = rng.random(flags.shape) < rate
    return np.where(mask, 1.0 - flags, flags)


def uniform_crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Each gene independently inherited from one parent or the other
    (50/50) -- unlike strategy.py's row-coupled crossover, gto_flags genes
    don't need to stay coherent with each other (each spot's flag is an
    independent on/off choice), so plain per-gene uniform crossover is the
    right (and simplest) operator here."""
    from_a = rng.random(a.shape) < 0.5
    return np.where(from_a, a, b)


class Genome:
    """num_buckets/thresholds: (strategy.NUM_BUCKETABLE,) / (..., MAX_BUCKETS-1)
    -- this genome's shared bucketing scheme for every non-boolean condition
    feature (see strategy.py).
    condition_features/condition_buckets: (strategy.NUM_STREETS,
    strategy.RULES_PER_STREET, strategy.CONDITIONS_PER_RULE) -- each rule's
    up-to-3 (feature, required bucket) conditions, indexed [street][rule]
    (street: 0=preflop, 1=flop, 2=turn, 3=river, matching Situation.street);
    condition_features entries are strategy.WILDCARD for an inactive ("don't
    care") slot. A stored bucket that its feature's current num_buckets
    doesn't have is clipped down to the highest one that does exist rather
    than ever left dangling (see strategy.effective_condition_bucket).
    rule_actions/rule_mix_actions: (strategy.NUM_STREETS,
    strategy.RULES_PER_STREET) -- each rule's action (index into
    strategy.ACTION_CATEGORIES -- Fold, Call, one of 6 fixed-size Raises, or
    All-In; each rule picks its own raise size independently). If
    rule_mix_actions is strategy.NO_MIX, the rule always plays rule_actions;
    otherwise it's a "Mix" -- a 50/50 coin flip between rule_actions and
    rule_mix_actions every time it fires (see strategy.resolve_rule_action).
    preflop_hole_category_mask: (strategy.RULES_PER_STREET,
    strategy.NUM_HOLE_CATEGORIES) -- mandatory for every rule in the preflop
    pool (street index strategy.PREFLOP) only: which of the 12
    hole-hand-category buckets that rule applies to (1.0 = applies, 0.0 =
    doesn't). Checked exactly against the current hand's hole hand category,
    not through the usual bucket-threshold mechanism -- see strategy.py.
    gto_flags: unchanged from before -- see module docstring.
    """

    __slots__ = (
        "num_buckets", "thresholds",
        "condition_features", "condition_buckets", "rule_actions", "rule_mix_actions",
        "preflop_hole_category_mask",
        "gto_flags",
    )

    def __init__(
        self,
        num_buckets: np.ndarray, thresholds: np.ndarray,
        condition_features: np.ndarray, condition_buckets: np.ndarray,
        rule_actions: np.ndarray, rule_mix_actions: np.ndarray,
        preflop_hole_category_mask: np.ndarray,
        gto_flags: np.ndarray,
    ):
        self.num_buckets = num_buckets
        self.thresholds = thresholds
        self.condition_features = condition_features
        self.condition_buckets = condition_buckets
        self.rule_actions = rule_actions
        self.rule_mix_actions = rule_mix_actions
        self.preflop_hole_category_mask = preflop_hole_category_mask
        self.gto_flags = gto_flags

    @classmethod
    def random(cls, rng: np.random.Generator, scale: float = 0.5) -> "Genome":
        """`scale` (relative to its old default of 0.5) scales how many of a
        freshly random genome's rule conditions start active rather than
        wildcard -- there's no "weight magnitude" analog to scale in this
        representation, but this preserves scale's role as an initial-
        complexity knob."""
        num_buckets = rng.integers(strategy.MIN_BUCKETS, strategy.MAX_BUCKETS + 1, size=strategy.NUM_BUCKETABLE)
        thresholds = np.sort(rng.random((strategy.NUM_BUCKETABLE, strategy.MAX_BUCKETS - 1)), axis=-1)

        active_prob = float(np.clip(strategy.CONDITION_ACTIVE_INIT_PROB * (scale / 0.5), 0.0, 1.0))
        per_street_shape = (strategy.RULES_PER_STREET, strategy.CONDITIONS_PER_RULE)
        # Each street draws its random feature choices from its own eligible
        # pool (preflop excludes board/made-hand features that need a board
        # that doesn't exist yet -- see strategy.ELIGIBLE_CONDITION_INDICES_BY_STREET).
        random_features = np.stack([
            rng.choice(strategy.ELIGIBLE_CONDITION_INDICES_BY_STREET[street], size=per_street_shape)
            for street in range(strategy.NUM_STREETS)
        ])
        is_active = rng.random(random_features.shape) < active_prob
        condition_features = np.where(is_active, random_features, strategy.WILDCARD)
        condition_buckets = rng.integers(0, strategy.MAX_BUCKETS, size=random_features.shape)
        # A rule with every slot wildcard is indistinguishable from the
        # pool's own trailing default -- never a meaningful thing for a rule
        # itself to be (see strategy.enforce_min_one_condition).
        for street in range(strategy.NUM_STREETS):
            condition_features[street], condition_buckets[street], _ = strategy.enforce_min_one_condition(
                condition_features[street], condition_buckets[street],
                strategy.ELIGIBLE_CONDITION_INDICES_BY_STREET[street], rng,
            )

        action_shape = (strategy.NUM_STREETS, strategy.RULES_PER_STREET)
        rule_actions = rng.integers(0, strategy.NUM_ACTION_CATEGORIES, size=action_shape)
        is_mix = rng.random(action_shape) < strategy.MIX_INIT_PROB
        random_mix = rng.integers(0, strategy.NUM_ACTION_CATEGORIES, size=action_shape)
        rule_mix_actions = np.where(is_mix, random_mix, strategy.NO_MIX)

        preflop_hole_category_mask = (
            rng.random((strategy.RULES_PER_STREET, strategy.NUM_HOLE_CATEGORIES)) < strategy.HOLE_CATEGORY_INIT_PROB
        ).astype(np.float64)

        return cls(
            num_buckets=num_buckets, thresholds=thresholds,
            condition_features=condition_features, condition_buckets=condition_buckets,
            rule_actions=rule_actions, rule_mix_actions=rule_mix_actions,
            preflop_hole_category_mask=preflop_hole_category_mask,
            gto_flags=(rng.random(NUM_GTO_SPOTS) < GTO_INIT_PROB).astype(np.float64),
        )

    def to_dict(self) -> dict:
        """Named-dictionary save form -- keyed by feature/GTO-spot/street
        name rather than array position, so a saved genome survives
        features.py/gto.py entries being added, removed, or reordered (see
        from_dict, the corresponding loader)."""
        feature_buckets = {}
        for row, idx in enumerate(strategy.BUCKETABLE_INDICES):
            spec = strategy.CONDITION_FEATURES[idx]
            feature_buckets[spec.key] = {
                "num_buckets": int(self.num_buckets[row]),
                "thresholds": [float(x) for x in self.thresholds[row]],
            }

        rules = {}
        for street, street_label in enumerate(strategy.STREET_LABELS):
            street_rules = []
            for r in range(strategy.RULES_PER_STREET):
                conditions = []
                for c in range(strategy.CONDITIONS_PER_RULE):
                    fi = int(self.condition_features[street, r, c])
                    conditions.append({
                        "feature": strategy.CONDITION_FEATURES[fi].key if fi != strategy.WILDCARD else None,
                        "bucket": int(self.condition_buckets[street, r, c]),
                    })
                mix_idx = int(self.rule_mix_actions[street, r])
                rule = {
                    "conditions": conditions,
                    "action": strategy.ACTION_CATEGORIES[int(self.rule_actions[street, r])],
                    "mix_action": strategy.ACTION_CATEGORIES[mix_idx] if mix_idx != strategy.NO_MIX else None,
                }
                if street == strategy.PREFLOP:
                    rule["hole_categories"] = [
                        label for label, flag in zip(strategy.HOLE_CATEGORY_LABELS, self.preflop_hole_category_mask[r])
                        if flag > 0.5
                    ]
                street_rules.append(rule)
            rules[street_label] = street_rules

        return {
            "feature_buckets": feature_buckets,
            "rules": rules,
            "gto_flags": {spot.key: float(flag) for spot, flag in zip(GTO_SPOTS, self.gto_flags)},
        }

    @classmethod
    def from_dict(cls, data: dict, rng: np.random.Generator) -> "Genome":
        """Reconstructs a Genome from its named-dictionary save form (see
        to_dict). Robust to the feature/GTO-spot catalog (or RULES_PER_STREET/
        CONDITIONS_PER_RULE) having changed since this genome was saved:
        entries whose name/shape no longer exists are dropped or truncated
        (with a warning); entries the current catalog expects but the save
        doesn't have are freshly initialized exactly as a brand-new random
        genome would be (with a warning), rather than defaulted to some
        placeholder value the GA never actually selected for. A save from
        before rules were split per street (a flat list rather than a
        per-street dict) can't be meaningfully mapped onto the new pools, so
        it's treated as "no rules saved" rather than misassigned."""
        saved_buckets = data.get("feature_buckets", {})
        known_keys = {spec.key for spec in strategy.CONDITION_FEATURES if spec.kind != "boolean"}
        unknown_features = sorted(set(saved_buckets) - known_keys)
        if unknown_features:
            print(
                f"Warning: ignoring {len(unknown_features)} unknown feature(s) in saved genome's "
                f"'feature_buckets' (no longer in the feature catalog): {', '.join(unknown_features)}"
            )
        missing_features = sorted(known_keys - set(saved_buckets))
        if missing_features:
            print(
                f"Warning: {len(missing_features)} feature(s) missing from saved genome's "
                f"'feature_buckets' (new since this genome was saved) -- initializing randomly: "
                f"{', '.join(missing_features)}"
            )
        num_buckets = np.empty(strategy.NUM_BUCKETABLE, dtype=np.int64)
        thresholds = np.empty((strategy.NUM_BUCKETABLE, strategy.MAX_BUCKETS - 1), dtype=np.float64)
        for row, idx in enumerate(strategy.BUCKETABLE_INDICES):
            key = strategy.CONDITION_FEATURES[idx].key
            if key in saved_buckets:
                entry = saved_buckets[key]
                num_buckets[row] = int(entry["num_buckets"])
                vals = [float(x) for x in entry["thresholds"]]
                thresholds[row] = (vals + [0.0] * strategy.MAX_BUCKETS)[: strategy.MAX_BUCKETS - 1]
            else:
                num_buckets[row] = rng.integers(strategy.MIN_BUCKETS, strategy.MAX_BUCKETS + 1)
                thresholds[row] = np.sort(rng.random(strategy.MAX_BUCKETS - 1))

        saved_rules_by_street = data.get("rules", {})
        # Older save formats (before rules were split per street) used a
        # flat list -- there's no sane way to guess which street each of
        # those rules belonged to, so treat it the same as "no rules saved
        # for any street" (freshly random) rather than misassigning them.
        if not isinstance(saved_rules_by_street, dict):
            saved_rules_by_street = {}

        condition_shape = (strategy.NUM_STREETS, strategy.RULES_PER_STREET, strategy.CONDITIONS_PER_RULE)
        condition_features = np.full(condition_shape, strategy.WILDCARD, dtype=np.int64)
        condition_buckets = np.zeros(condition_shape, dtype=np.int64)
        action_shape = (strategy.NUM_STREETS, strategy.RULES_PER_STREET)
        rule_actions = np.zeros(action_shape, dtype=np.int64)
        rule_mix_actions = np.full(action_shape, strategy.NO_MIX, dtype=np.int64)
        preflop_hole_category_mask = np.zeros((strategy.RULES_PER_STREET, strategy.NUM_HOLE_CATEGORIES), dtype=np.float64)
        hole_category_index_by_label = {label: i for i, label in enumerate(strategy.HOLE_CATEGORY_LABELS)}
        action_index_by_name = {name: i for i, name in enumerate(strategy.ACTION_CATEGORIES)}
        unknown_rule_features = set()
        unknown_hole_categories = set()
        ineligible_preflop_features = set()
        repaired_no_condition_by_street: dict[str, int] = {}

        for street, street_label in enumerate(strategy.STREET_LABELS):
            saved_rules = saved_rules_by_street.get(street_label, [])
            if len(saved_rules) != strategy.RULES_PER_STREET:
                print(
                    f"Warning: saved genome has {len(saved_rules)} {street_label} rule(s), current "
                    f"RULES_PER_STREET is {strategy.RULES_PER_STREET} -- truncating or padding with "
                    "fresh random rules."
                )
            for r in range(strategy.RULES_PER_STREET):
                if r < len(saved_rules):
                    rule = saved_rules[r]
                    for c, cond in enumerate(rule.get("conditions", [])[: strategy.CONDITIONS_PER_RULE]):
                        feature_key = cond.get("feature")
                        condition_buckets[street, r, c] = int(cond.get("bucket", 0))
                        if feature_key is None:
                            condition_features[street, r, c] = strategy.WILDCARD
                            continue
                        try:
                            fi = strategy.feature_index(feature_key)
                        except KeyError:
                            unknown_rule_features.add(feature_key)
                            condition_features[street, r, c] = strategy.WILDCARD
                            continue
                        # Some features are meaningless preflop (no board yet, or
                        # nothing to compare the hole cards against) -- a save
                        # containing one there (e.g. from before this restriction
                        # existed) has it dropped, same as an unknown one.
                        if street == strategy.PREFLOP and fi not in strategy.ELIGIBLE_CONDITION_INDICES_BY_STREET[street]:
                            ineligible_preflop_features.add(feature_key)
                            condition_features[street, r, c] = strategy.WILDCARD
                            continue
                        condition_features[street, r, c] = fi
                    rule_actions[street, r] = action_index_by_name.get(rule.get("action"), strategy.ACTION_FOLD)
                    rule_mix_actions[street, r] = action_index_by_name.get(rule.get("mix_action"), strategy.NO_MIX)
                    if street == strategy.PREFLOP:
                        for label in rule.get("hole_categories", []):
                            if label in hole_category_index_by_label:
                                preflop_hole_category_mask[r, hole_category_index_by_label[label]] = 1.0
                            else:
                                unknown_hole_categories.add(label)
                        if "hole_categories" not in rule:
                            print(
                                f"Warning: Preflop rule {r} in saved genome has no 'hole_categories' -- "
                                "initializing randomly."
                            )
                            preflop_hole_category_mask[r] = (
                                rng.random(strategy.NUM_HOLE_CATEGORIES) < strategy.HOLE_CATEGORY_INIT_PROB
                            )
                else:
                    eligible = strategy.ELIGIBLE_CONDITION_INDICES_BY_STREET[street]
                    random_features = rng.choice(eligible, size=strategy.CONDITIONS_PER_RULE)
                    is_active = rng.random(strategy.CONDITIONS_PER_RULE) < strategy.CONDITION_ACTIVE_INIT_PROB
                    condition_features[street, r] = np.where(is_active, random_features, strategy.WILDCARD)
                    condition_buckets[street, r] = rng.integers(0, strategy.MAX_BUCKETS, size=strategy.CONDITIONS_PER_RULE)
                    rule_actions[street, r] = rng.integers(0, strategy.NUM_ACTION_CATEGORIES)
                    rule_mix_actions[street, r] = (
                        int(rng.integers(0, strategy.NUM_ACTION_CATEGORIES))
                        if rng.random() < strategy.MIX_INIT_PROB else strategy.NO_MIX
                    )
                    if street == strategy.PREFLOP:
                        preflop_hole_category_mask[r] = (
                            rng.random(strategy.NUM_HOLE_CATEGORIES) < strategy.HOLE_CATEGORY_INIT_PROB
                        )
            # A saved rule with no conditions at all (e.g. from before this
            # constraint existed) is indistinguishable from the pool's own
            # trailing default -- repaired the same way a freshly random or
            # freshly mutated all-wildcard rule is (see
            # strategy.enforce_min_one_condition).
            condition_features[street], condition_buckets[street], num_repaired = strategy.enforce_min_one_condition(
                condition_features[street], condition_buckets[street],
                strategy.ELIGIBLE_CONDITION_INDICES_BY_STREET[street], rng,
            )
            if num_repaired:
                repaired_no_condition_by_street[street_label] = num_repaired
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
        if repaired_no_condition_by_street:
            total_repaired = sum(repaired_no_condition_by_street.values())
            detail = ", ".join(f"{n} {label}" for label, n in repaired_no_condition_by_street.items())
            print(
                f"Warning: {total_repaired} rule(s) in saved genome had no conditions at all -- adding one "
                f"random condition to each ({detail})."
            )
        if unknown_hole_categories:
            print(
                f"Warning: ignoring {len(unknown_hole_categories)} unknown hole hand category label(s) in "
                f"saved genome's preflop rules (no longer in the catalog): {', '.join(sorted(unknown_hole_categories))}"
            )

        saved_gto = data.get("gto_flags", {})
        gto_keys = [spot.key for spot in GTO_SPOTS]
        unknown_gto = sorted(set(saved_gto) - set(gto_keys))
        if unknown_gto:
            print(
                f"Warning: ignoring {len(unknown_gto)} unknown GTO spot(s) in saved genome "
                f"(no longer in the catalog): {', '.join(unknown_gto)}"
            )
        missing_gto = [key for key in gto_keys if key not in saved_gto]
        if missing_gto:
            print(
                f"Warning: {len(missing_gto)} GTO spot(s) missing from saved genome (new "
                f"since this genome was saved) -- initializing randomly: {', '.join(missing_gto)}"
            )
        gto_flags = np.array(
            [
                float(saved_gto[key]) if key in saved_gto else float(rng.random() < GTO_INIT_PROB)
                for key in gto_keys
            ],
            dtype=np.float64,
        )

        return cls(
            num_buckets, thresholds, condition_features, condition_buckets,
            rule_actions, rule_mix_actions,
            preflop_hole_category_mask, gto_flags,
        )

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str, rng: np.random.Generator | None = None) -> "Genome":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data, rng if rng is not None else np.random.default_rng())

    def copy(self) -> "Genome":
        return Genome(
            self.num_buckets.copy(), self.thresholds.copy(),
            self.condition_features.copy(), self.condition_buckets.copy(),
            self.rule_actions.copy(), self.rule_mix_actions.copy(),
            self.preflop_hole_category_mask.copy(),
            self.gto_flags.copy(),
        )

    def mutate(self, rng: np.random.Generator, rate: float, continuous_scale: float) -> "Genome":
        """Returns a mutated copy. Each gene kind gets the operator that
        matches its structure (see strategy.py for the reasoning behind
        each): num_buckets bit-flips between 2/3; thresholds get additive-
        gaussian noise rescaled for their 0-1 domain (see
        strategy.THRESHOLD_MUTATION_SCALE_FACTOR); condition features get a
        full random reassignment (no meaningful "nudge" for feature
        identity), restricted per street to that street's eligible pool
        (preflop excludes features that need a board -- see
        strategy.ELIGIBLE_CONDITION_INDICES_BY_STREET); condition buckets
        and rule_actions get the nudge-mostly/jump-sometimes blend, since
        those integers *are* meaningfully ordered (rule actions run
        Check/Fold < Call < Raise 25% < ... < Raise 150% < All-In, so a
        nudge means "slightly more/less aggressive, or a slightly bigger/
        smaller raise"); rule_mix_actions gets a full random reassignment
        like condition features (turning a rule into/out of a Mix isn't an
        incremental move); preflop_hole_category_mask and gto_flags
        bit-flip (same operator, reused -- there's nothing "in between" a
        rule claiming a hole category or a genome trusting a GTO chart,
        same as any other boolean gene). Each gene is independently
        selected for mutation with probability `rate`, whichever kind it is.
        A repair pass afterward restores any rule mutation reduced to all-
        wildcard back to having at least one condition (see
        strategy.enforce_min_one_condition)."""
        threshold_scale = continuous_scale * strategy.THRESHOLD_MUTATION_SCALE_FACTOR

        condition_buckets = strategy.mutate_condition_buckets(self.condition_buckets, rate, rng)
        condition_features = []
        for street in range(strategy.NUM_STREETS):
            mutated_features = strategy.mutate_condition_features(
                self.condition_features[street], rate, rng, strategy.ELIGIBLE_CONDITION_INDICES_BY_STREET[street],
            )
            repaired_features, repaired_buckets, _ = strategy.enforce_min_one_condition(
                mutated_features, condition_buckets[street], strategy.ELIGIBLE_CONDITION_INDICES_BY_STREET[street], rng,
            )
            condition_features.append(repaired_features)
            condition_buckets[street] = repaired_buckets
        condition_features = np.stack(condition_features)

        return Genome(
            strategy.mutate_num_buckets(self.num_buckets, rate, rng),
            strategy.mutate_thresholds(self.thresholds, rate, threshold_scale, rng),
            condition_features,
            condition_buckets,
            strategy.mutate_rule_actions(self.rule_actions, rate, rng),
            strategy.mutate_rule_mix_actions(self.rule_mix_actions, rate, rng),
            mutate_bool_flags(self.preflop_hole_category_mask, rate, rng),
            mutate_bool_flags(self.gto_flags, rate, rng),
        )

    def crossover(self, other: "Genome", rng: np.random.Generator) -> "Genome":
        """Returns a child combining self and other. Two coupled row-crossovers
        (see strategy.apply_row_mask) keep gene groups that must stay
        internally coherent inherited from the *same* parent: a feature's
        num_buckets always travels with its own thresholds (never a bucket
        count from one parent paired with cut points sized for a different
        count), and -- across every street's pool at once, one coin flip per
        (street, rule) slot -- a rule's conditions, action, mix action, and
        (for preflop) hole_category_mask always travel together (never a
        feature index from one parent matched against an unrelated bucket
        index from the other, or a hole-category claim that belonged to a
        different rule's conditions). gto_flags are independent per-gene
        choices, so they use plain uniform crossover."""
        feature_mask = strategy.row_crossover_mask(strategy.NUM_BUCKETABLE, rng)
        num_buckets = strategy.apply_row_mask(self.num_buckets, other.num_buckets, feature_mask)
        thresholds = strategy.apply_row_mask(self.thresholds, other.thresholds, feature_mask)

        num_rule_slots = strategy.NUM_STREETS * strategy.RULES_PER_STREET
        rule_mask_flat = strategy.row_crossover_mask(num_rule_slots, rng)

        condition_shape = self.condition_features.shape
        flat_condition_shape = (num_rule_slots, strategy.CONDITIONS_PER_RULE)
        condition_features = strategy.apply_row_mask(
            self.condition_features.reshape(flat_condition_shape),
            other.condition_features.reshape(flat_condition_shape),
            rule_mask_flat,
        ).reshape(condition_shape)
        condition_buckets = strategy.apply_row_mask(
            self.condition_buckets.reshape(flat_condition_shape),
            other.condition_buckets.reshape(flat_condition_shape),
            rule_mask_flat,
        ).reshape(condition_shape)
        rule_actions = strategy.apply_row_mask(
            self.rule_actions.reshape(num_rule_slots),
            other.rule_actions.reshape(num_rule_slots),
            rule_mask_flat,
        ).reshape(self.rule_actions.shape)
        rule_mix_actions = strategy.apply_row_mask(
            self.rule_mix_actions.reshape(num_rule_slots),
            other.rule_mix_actions.reshape(num_rule_slots),
            rule_mask_flat,
        ).reshape(self.rule_mix_actions.shape)

        preflop_rule_mask = rule_mask_flat.reshape(strategy.NUM_STREETS, strategy.RULES_PER_STREET)[strategy.PREFLOP]
        preflop_hole_category_mask = strategy.apply_row_mask(
            self.preflop_hole_category_mask, other.preflop_hole_category_mask, preflop_rule_mask,
        )

        return Genome(
            num_buckets, thresholds, condition_features, condition_buckets,
            rule_actions, rule_mix_actions,
            preflop_hole_category_mask,
            uniform_crossover(self.gto_flags, other.gto_flags, rng),
        )

    def nonzero_weight_count(self) -> int:
        """Number of active (non-wildcard) rule conditions across the whole
        strategy -- how many (feature, threshold) facts a human needs to
        memorize to execute this genome's strategy. The rule-based successor
        to the old linear system's "how many nonzero weights" proxy; kept
        under the same method name since main.py's sparsity penalty and
        tournament.py's leaderboard call it by name without caring about the
        representation. Doesn't include gto_flags -- a different kind of
        complexity (how many memorized charts, not how many rule
        conditions)."""
        return int(np.count_nonzero(self.condition_features != strategy.WILDCARD))

    def active_gto_spots(self) -> list:
        """The GTO_SPOTS entries this genome currently trusts (gto_flags is
        on), in catalog order -- the same order decide() checks them in."""
        return [spot for spot, flag in zip(GTO_SPOTS, self.gto_flags) if flag > 0.5]

    def force_all_gto_active(self) -> None:
        """In-place: sets every gto_flags gene to active, overriding whatever
        mutation/crossover produced. Used by ga.py's IslandModel to keep
        --force-gto-islands islands always trusting every GTO_SPOTS chart
        instead of letting gto_flags evolve freely."""
        self.gto_flags = np.ones(NUM_GTO_SPOTS, dtype=np.float64)

    def decide(
        self,
        situation: Situation,
        legal_actions: list[int],
        rng: np.random.Generator | None = None,
    ) -> tuple[int, float]:
        """Returns (action, raw_bet_size_in_chips_if_betting_else_0)."""
        gto_result = self._decide_from_gto_charts(situation, legal_actions)
        if gto_result is not None:
            return gto_result

        features = extract_features(situation)
        condition_values = features[strategy.CONDITION_FEATURES_FULL_INDEX]
        buckets = strategy.compute_all_buckets(condition_values, self.num_buckets, self.thresholds)

        street = situation.street
        if street == strategy.PREFLOP:
            hole_category = strategy.hole_category_index(condition_values)
            rule_idx = strategy.first_matching_preflop_rule_index(
                buckets, self.condition_features[street], self.condition_buckets[street], self.num_buckets,
                self.preflop_hole_category_mask, hole_category,
            )
        else:
            rule_idx = strategy.first_matching_rule_index(
                buckets, self.condition_features[street], self.condition_buckets[street], self.num_buckets,
            )

        if rule_idx is None:
            action_idx = strategy.ACTION_FOLD  # no rule matched: default to Fold, like a real chart's blank squares
        else:
            action_idx = strategy.resolve_rule_action(
                int(self.rule_actions[street, rule_idx]), int(self.rule_mix_actions[street, rule_idx]), rng,
            )

        if action_idx == strategy.ACTION_FOLD:
            action = FOLD if FOLD in legal_actions else CHECK_CALL
            return action, 0.0
        if action_idx == strategy.ACTION_CALL:
            return CHECK_CALL, 0.0

        # One of the 6 Raise categories, or ACTION_ALLIN
        if BET_RAISE not in legal_actions:
            return CHECK_CALL, 0.0
        if action_idx == strategy.ACTION_ALLIN:
            bet_size = situation.my_stack
        else:
            bet_size = strategy.RAISE_POT_FRACTION[action_idx] * max(situation.pot, 1.0)
        return BET_RAISE, bet_size

    def _decide_from_gto_charts(
        self, situation: Situation, legal_actions: list[int],
    ) -> tuple[int, float] | None:
        """Checks this genome's active GTO_SPOTS (gto_flags on), in catalog
        order, for the first one whose spot matches `situation` -- if found,
        plays exactly what that spot's chart says for this hand, bypassing
        the rule list entirely. Returns None if no active spot applies, so
        decide() falls through to the normal rule-based decision."""
        for i, spot in enumerate(GTO_SPOTS):
            if self.gto_flags[i] <= 0.5:
                continue
            resolved = resolve_spot_action(spot, situation)
            if resolved is None:
                continue
            kind, size_spec = resolved
            if kind == "fold":
                action = FOLD if FOLD in legal_actions else CHECK_CALL
                return action, 0.0
            if kind == "call":
                return CHECK_CALL, 0.0
            # kind == "raise"
            if BET_RAISE not in legal_actions:
                return CHECK_CALL, 0.0
            if size_spec is None:
                bet_size = situation.my_stack  # "allin": shove the full stack
            elif size_spec[0] == "pot":
                bet_size = size_spec[1] * max(situation.pot, 1.0)
            else:  # ("bb", n): raise so my total commitment this street reaches n big blinds.
                # call_amount == current_bet for every non-blind seat that hasn't
                # committed anything yet this street (the common case these
                # spots are written for), so this is exact there and a close
                # approximation for the blinds.
                target_total = size_spec[1] * situation.big_blind
                bet_size = max(target_total - situation.call_amount, 0.0)
            return BET_RAISE, bet_size
        return None


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
