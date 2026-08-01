"""The evolvable unit: a small, ordered list of "if this situation, then
this action" rules -- the kind of simplified strategy a human could actually
memorize and execute at a table, rather than doing live arithmetic.

This replaces an earlier linear "V/L/theta" system (two 0-100 scores per
decision, combined through evolvable thresholds) that was accurate but not
executable by a human: reading it required tracking which of ~200 weighted
features were active and summing them live, every decision, every hand.

The new decision rule, in full (see strategy.py for the machinery this
delegates to):

  1. Bucket every "top-level" feature (the ~50 generalized concepts in
     features.py, not its ~150 one-hot indicator children) into 2-3 groups
     via this genome's own evolved thresholds -- shared across every rule,
     so e.g. "Hand Strength: weak/medium/strong" has one definition the
     whole strategy reads from, the way a real chart would.
  2. Check this genome's NUM_RULES rules in fixed order; each is a small
     conjunction of up to CONDITIONS_PER_RULE (feature, required bucket)
     checks (or a wildcard "don't care"). First full match wins -- the same
     idiom gto.py's GTOSpot.action_ranges/SpotMatcher already use.
  3. That rule's action category -- Fold / Call / Raise (to one shared,
     evolved pot-fraction size) / All-In -- is what gets played. No match at
     all defaults to Fold (mirrors GTOSpot.default_action's "everything not
     colored in is a fold").

Every decision re-evaluates this list fresh from the current Situation --
there's no memory of "I'm currently bluffing" carried between streets.
"Raise now, fold if raised back" isn't a thing this system represents
explicitly: it just falls out of a *later* decision's bigger call_amount/
facing_bet naturally routing to a different (likely Fold) rule. Likewise
"call up to X, else fold" is just an ordinary rule condition on the existing
call_amount_norm feature's bucket, not a special parameter -- once the bet
gets too big for that condition to match, control falls through to whatever
rule the genome evolved as its fallback (typically Fold).

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

# Init center/spread for bucket_noise_std -- small, since it lives on
# features.py's native 0-1 scale (jittering a normalized feature value, not
# a V/L-style percentile) and is meant to only nudge borderline calls near a
# threshold, not meaningfully blur the strategy.
BUCKET_NOISE_STD_INIT = (0.03, 0.03)  # (center, spread), then abs()


def mutate_bool_flags(flags: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Bit-flip mutation for boolean genes (0.0/1.0 floats): each selected
    gene flips to its opposite, rather than being nudged like a continuous
    value or alphabet-jumped like a categorical one -- there's nothing "in
    between" on or off. Used for gto_flags."""
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
    -- this genome's shared bucketing scheme for every non-boolean top-level
    feature (see strategy.py).
    condition_features/condition_buckets: (strategy.NUM_RULES,
    strategy.CONDITIONS_PER_RULE) -- each rule's up-to-3 (feature, required
    bucket) conditions; condition_features entries are strategy.WILDCARD for
    an inactive ("don't care") slot.
    rule_actions: (strategy.NUM_RULES,) -- each rule's action category
    (index into strategy.ACTION_CATEGORIES).
    raise_size_idx: index into strategy.RAISE_SIZE_ALPHABET -- the one
    shared pot-fraction size every Raise-category rule plays.
    bucket_noise_std: small Gaussian jitter applied to a feature's
    normalized value before bucketing it, giving cheap, natural mixed-
    strategy behavior right at a threshold boundary.
    gto_flags: unchanged from before -- see module docstring.
    """

    __slots__ = (
        "num_buckets", "thresholds",
        "condition_features", "condition_buckets", "rule_actions",
        "raise_size_idx", "bucket_noise_std",
        "gto_flags",
    )

    def __init__(
        self,
        num_buckets: np.ndarray, thresholds: np.ndarray,
        condition_features: np.ndarray, condition_buckets: np.ndarray, rule_actions: np.ndarray,
        raise_size_idx: int, bucket_noise_std: float,
        gto_flags: np.ndarray,
    ):
        self.num_buckets = num_buckets
        self.thresholds = thresholds
        self.condition_features = condition_features
        self.condition_buckets = condition_buckets
        self.rule_actions = rule_actions
        self.raise_size_idx = raise_size_idx
        self.bucket_noise_std = bucket_noise_std
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
        condition_shape = (strategy.NUM_RULES, strategy.CONDITIONS_PER_RULE)
        random_features = rng.integers(0, strategy.NUM_TOP_LEVEL_FEATURES, size=condition_shape)
        is_active = rng.random(condition_shape) < active_prob
        condition_features = np.where(is_active, random_features, strategy.WILDCARD)
        condition_buckets = rng.integers(0, strategy.MAX_BUCKETS, size=condition_shape)
        rule_actions = rng.integers(0, strategy.NUM_ACTION_CATEGORIES, size=strategy.NUM_RULES)

        raise_size_idx = int(rng.integers(0, len(strategy.RAISE_SIZE_ALPHABET)))
        center, spread = BUCKET_NOISE_STD_INIT
        bucket_noise_std = abs(float(rng.normal(center, spread)))

        return cls(
            num_buckets=num_buckets, thresholds=thresholds,
            condition_features=condition_features, condition_buckets=condition_buckets,
            rule_actions=rule_actions,
            raise_size_idx=raise_size_idx, bucket_noise_std=bucket_noise_std,
            gto_flags=(rng.random(NUM_GTO_SPOTS) < GTO_INIT_PROB).astype(np.float64),
        )

    def to_dict(self) -> dict:
        """Named-dictionary save form -- keyed by feature/GTO-spot name
        rather than array position, so a saved genome survives features.py/
        gto.py entries being added, removed, or reordered (see from_dict,
        the corresponding loader)."""
        feature_buckets = {}
        for row, idx in enumerate(strategy.BUCKETABLE_INDICES):
            spec = strategy.TOP_LEVEL_FEATURES[idx]
            feature_buckets[spec.key] = {
                "num_buckets": int(self.num_buckets[row]),
                "thresholds": [float(x) for x in self.thresholds[row]],
            }

        rules = []
        for r in range(strategy.NUM_RULES):
            conditions = []
            for c in range(strategy.CONDITIONS_PER_RULE):
                fi = int(self.condition_features[r, c])
                conditions.append({
                    "feature": strategy.TOP_LEVEL_FEATURES[fi].key if fi != strategy.WILDCARD else None,
                    "bucket": int(self.condition_buckets[r, c]),
                })
            rules.append({
                "conditions": conditions,
                "action": strategy.ACTION_CATEGORIES[int(self.rule_actions[r])],
            })

        return {
            "feature_buckets": feature_buckets,
            "rules": rules,
            "raise_size_pct": float(strategy.RAISE_SIZE_ALPHABET[self.raise_size_idx]),
            "bucket_noise_std": float(self.bucket_noise_std),
            "gto_flags": {spot.key: float(flag) for spot, flag in zip(GTO_SPOTS, self.gto_flags)},
        }

    @classmethod
    def from_dict(cls, data: dict, rng: np.random.Generator) -> "Genome":
        """Reconstructs a Genome from its named-dictionary save form (see
        to_dict). Robust to the feature/GTO-spot catalog (or NUM_RULES/
        CONDITIONS_PER_RULE) having changed since this genome was saved:
        entries whose name/shape no longer exists are dropped or truncated
        (with a warning); entries the current catalog expects but the save
        doesn't have are freshly initialized exactly as a brand-new random
        genome would be (with a warning), rather than defaulted to some
        placeholder value the GA never actually selected for."""
        saved_buckets = data.get("feature_buckets", {})
        known_keys = {spec.key for spec in strategy.TOP_LEVEL_FEATURES if spec.kind != "boolean"}
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
            key = strategy.TOP_LEVEL_FEATURES[idx].key
            if key in saved_buckets:
                entry = saved_buckets[key]
                num_buckets[row] = int(entry["num_buckets"])
                vals = [float(x) for x in entry["thresholds"]]
                thresholds[row] = (vals + [0.0] * strategy.MAX_BUCKETS)[: strategy.MAX_BUCKETS - 1]
            else:
                num_buckets[row] = rng.integers(strategy.MIN_BUCKETS, strategy.MAX_BUCKETS + 1)
                thresholds[row] = np.sort(rng.random(strategy.MAX_BUCKETS - 1))

        saved_rules = data.get("rules", [])
        if len(saved_rules) != strategy.NUM_RULES:
            print(
                f"Warning: saved genome has {len(saved_rules)} rule(s), current NUM_RULES is "
                f"{strategy.NUM_RULES} -- truncating or padding with fresh random rules."
            )
        condition_shape = (strategy.NUM_RULES, strategy.CONDITIONS_PER_RULE)
        condition_features = np.full(condition_shape, strategy.WILDCARD, dtype=np.int64)
        condition_buckets = np.zeros(condition_shape, dtype=np.int64)
        rule_actions = np.zeros(strategy.NUM_RULES, dtype=np.int64)
        action_index_by_name = {name: i for i, name in enumerate(strategy.ACTION_CATEGORIES)}
        unknown_rule_features = set()
        for r in range(strategy.NUM_RULES):
            if r < len(saved_rules):
                rule = saved_rules[r]
                for c, cond in enumerate(rule.get("conditions", [])[: strategy.CONDITIONS_PER_RULE]):
                    feature_key = cond.get("feature")
                    condition_buckets[r, c] = int(cond.get("bucket", 0))
                    if feature_key is None:
                        condition_features[r, c] = strategy.WILDCARD
                        continue
                    try:
                        condition_features[r, c] = strategy.feature_index(feature_key)
                    except KeyError:
                        unknown_rule_features.add(feature_key)
                        condition_features[r, c] = strategy.WILDCARD
                rule_actions[r] = action_index_by_name.get(rule.get("action"), strategy.ACTION_FOLD)
            else:
                random_features = rng.integers(0, strategy.NUM_TOP_LEVEL_FEATURES, size=strategy.CONDITIONS_PER_RULE)
                is_active = rng.random(strategy.CONDITIONS_PER_RULE) < strategy.CONDITION_ACTIVE_INIT_PROB
                condition_features[r] = np.where(is_active, random_features, strategy.WILDCARD)
                condition_buckets[r] = rng.integers(0, strategy.MAX_BUCKETS, size=strategy.CONDITIONS_PER_RULE)
                rule_actions[r] = rng.integers(0, strategy.NUM_ACTION_CATEGORIES)
        if unknown_rule_features:
            print(
                f"Warning: ignoring {len(unknown_rule_features)} unknown feature(s) referenced by saved "
                f"genome's rules (no longer in the feature catalog): {', '.join(sorted(unknown_rule_features))}"
            )

        if "raise_size_pct" in data:
            saved_pct = float(data["raise_size_pct"])
            raise_size_idx = int(np.argmin(np.abs(strategy.RAISE_SIZE_ALPHABET - saved_pct)))
        else:
            print("Warning: 'raise_size_pct' missing from saved genome -- initializing randomly.")
            raise_size_idx = int(rng.integers(0, len(strategy.RAISE_SIZE_ALPHABET)))

        if "bucket_noise_std" in data:
            bucket_noise_std = float(data["bucket_noise_std"])
        else:
            print("Warning: 'bucket_noise_std' missing from saved genome -- initializing randomly.")
            center, spread = BUCKET_NOISE_STD_INIT
            bucket_noise_std = abs(float(rng.normal(center, spread)))

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
            num_buckets, thresholds, condition_features, condition_buckets, rule_actions,
            raise_size_idx, bucket_noise_std, gto_flags,
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
            self.condition_features.copy(), self.condition_buckets.copy(), self.rule_actions.copy(),
            self.raise_size_idx, self.bucket_noise_std,
            self.gto_flags.copy(),
        )

    def mutate(self, rng: np.random.Generator, rate: float, continuous_scale: float) -> "Genome":
        """Returns a mutated copy. Each gene kind gets the operator that
        matches its structure (see strategy.py for the reasoning behind
        each): num_buckets bit-flips between 2/3; thresholds and
        bucket_noise_std get additive-gaussian noise rescaled for their 0-1
        domain (see strategy.THRESHOLD_MUTATION_SCALE_FACTOR); condition
        features get a full random reassignment (no meaningful "nudge" for
        feature identity); condition buckets, rule actions, and
        raise_size_idx get the nudge-mostly/jump-sometimes blend, since
        those integers *are* meaningfully ordered; gto_flags bit-flip.
        Each gene is independently selected for mutation with probability
        `rate`, whichever kind it is."""
        threshold_scale = continuous_scale * strategy.THRESHOLD_MUTATION_SCALE_FACTOR

        def mutate_noise_std(value: float) -> float:
            if rng.random() < rate:
                value = value + float(rng.normal(0, threshold_scale))
            return abs(value)

        return Genome(
            strategy.mutate_num_buckets(self.num_buckets, rate, rng),
            strategy.mutate_thresholds(self.thresholds, rate, threshold_scale, rng),
            strategy.mutate_condition_features(self.condition_features, rate, rng),
            strategy.mutate_condition_buckets(self.condition_buckets, rate, rng),
            strategy.mutate_rule_actions(self.rule_actions, rate, rng),
            strategy.mutate_alphabet_index(self.raise_size_idx, len(strategy.RAISE_SIZE_ALPHABET), rate, rng),
            mutate_noise_std(self.bucket_noise_std),
            mutate_bool_flags(self.gto_flags, rate, rng),
        )

    def crossover(self, other: "Genome", rng: np.random.Generator) -> "Genome":
        """Returns a child combining self and other. Two coupled row-crossovers
        (see strategy.apply_row_mask) keep gene groups that must stay
        internally coherent inherited from the *same* parent: a feature's
        num_buckets always travels with its own thresholds (never a bucket
        count from one parent paired with cut points sized for a different
        count), and a rule's conditions always travel with its own action
        (never a feature index from one parent matched against an unrelated
        bucket index from the other). raise_size_idx and gto_flags are
        independent single/per-gene choices, so they use plain uniform
        crossover; bucket_noise_std is a genuinely continuous scalar, so it
        keeps blend crossover."""
        feature_mask = strategy.row_crossover_mask(strategy.NUM_BUCKETABLE, rng)
        num_buckets = strategy.apply_row_mask(self.num_buckets, other.num_buckets, feature_mask)
        thresholds = strategy.apply_row_mask(self.thresholds, other.thresholds, feature_mask)

        rule_mask = strategy.row_crossover_mask(strategy.NUM_RULES, rng)
        condition_features = strategy.apply_row_mask(self.condition_features, other.condition_features, rule_mask)
        condition_buckets = strategy.apply_row_mask(self.condition_buckets, other.condition_buckets, rule_mask)
        rule_actions = strategy.apply_row_mask(self.rule_actions, other.rule_actions, rule_mask)

        raise_size_idx = self.raise_size_idx if rng.random() < 0.5 else other.raise_size_idx

        alpha = rng.uniform(0.0, 1.0)
        bucket_noise_std = abs(alpha * self.bucket_noise_std + (1 - alpha) * other.bucket_noise_std)

        return Genome(
            num_buckets, thresholds, condition_features, condition_buckets, rule_actions,
            raise_size_idx, bucket_noise_std,
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
        top_level_values = features[strategy.TOP_LEVEL_FULL_INDEX]
        buckets = strategy.compute_all_buckets(
            top_level_values, self.num_buckets, self.thresholds, self.bucket_noise_std, rng,
        )
        action_idx = strategy.first_matching_rule(
            buckets, self.condition_features, self.condition_buckets, self.rule_actions,
        )
        if action_idx is None:
            action_idx = strategy.ACTION_FOLD  # no rule matched: default to Fold, like a real chart's blank squares

        if action_idx == strategy.ACTION_FOLD:
            action = FOLD if FOLD in legal_actions else CHECK_CALL
            return action, 0.0
        if action_idx == strategy.ACTION_CALL:
            return CHECK_CALL, 0.0

        # ACTION_RAISE or ACTION_ALLIN
        if BET_RAISE not in legal_actions:
            return CHECK_CALL, 0.0
        if action_idx == strategy.ACTION_ALLIN:
            bet_size = situation.my_stack
        else:
            bet_size = float(strategy.RAISE_SIZE_ALPHABET[self.raise_size_idx]) * max(situation.pot, 1.0)
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
