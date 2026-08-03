# poker_ga

A Single Deep CFR implementation that trains a 6-max No-Limit Hold'em
strategy over the real game engine.

## How it works

- **Game engine** (`poker_ga/game.py`): a real 6-max NLHE implementation --
  blinds, four streets, all-in handling with correct multi-way side pots,
  and showdown. A seat's decision is delegated to `Player.genome`, any
  duck-typed object exposing `.decide(situation, legal_actions, rng) ->
  (action, bet_size)` -- in practice a trained `cfr_policy.DeepCFRPolicy`
  during real play, or a test double in tests.
- **Features** (`poker_ga/features.py`): turns a decision point
  (`Situation`) into ~200 named, human-readable feature values -- made-hand
  category, hole/board card ranks, street, pot odds, position, stack depth,
  draws, board texture, opponent tendencies, and more (see `FEATURE_SPECS`)
  -- each with a label, a precise description, and a group, for both the
  network's own feature selection and `cfr_explorer.py`'s display. Most
  characteristics are represented twice: one generalized 0-1 feature and a
  set of exact per-value indicator features, so a model can learn either a
  linear trend or a value-specific effect.
- **Seating** (`poker_ga/seating.py`): pure seat-arithmetic shared by the
  game engine and feature extraction -- blind positions, preflop action
  order, and standard seat-role naming (UTG/HJ/CO/BTN/SB/BB) that correctly
  collapses at smaller table sizes.
- **Action space** (`poker_ga/strategy.py`, `poker_ga/genome.py`,
  `poker_ga/rules.py`, `poker_ga/cfr_actions.py`): the network predicts over
  a fixed 9-category discrete action set (`strategy.ACTION_CATEGORIES` --
  Fold, Call, six fixed-size pot-fraction Raises, All-In). `cfr_actions.py`
  is the one place that translates between this category space and the game
  engine's 3-way `legal_actions` (Fold/Check-or-Call/Bet-or-Raise,
  `genome.py`'s `FOLD`/`CHECK_CALL`/`BET_RAISE`), shared by both training
  (`cfr_tree.py`) and inference (`cfr_policy.py`) so they can never drift
  apart, and by regret matching (turning predicted regrets into a mixed
  strategy over legal actions).
- **CFR features** (`poker_ga/cfr_features.py`): maps a configurable subset
  of `features.py`'s vocabulary to the positions the advantage network
  reads out of `extract_features()`'s output vector. Unlike a hand-built
  rule system, the network can be pointed at any subset of the ~200 keys,
  including the one-hot indicator children -- `DEFAULT_FEATURE_KEYS` uses
  the full vocabulary (minus opponent-tendency reads, which need session
  history a single-hand CFR traversal doesn't have).
- **CFR tree traversal** (`poker_ga/cfr_tree.py`): external-sampling MCCFR
  over the real multiway betting mechanics, with each decision conditioned
  on the abstracted feature vector instead of a true information set. The
  traversing player branches into every legal action to compute
  counterfactual regret; every other seat samples one action from its
  current strategy (external sampling -- one shared net for every seat,
  genuinely self-play). A terminal showdown before the river is scored as
  an *equity* estimate over possible board completions, not one high-
  variance sampled runout, since Single Deep CFR's regression target is
  exactly that counterfactual value. Every traversed hand independently
  redraws each seat's own starting stack, uniformly between
  `--min-starting-stack-bb`/`--max-starting-stack-bb` (default 20-200BB) --
  a real multi-hand tournament session has players sitting at a whole
  spread of stack depths at once (short after losing pots, deep after
  winning them), not everyone re-buying to the same depth every hand, so
  training only ever at one fixed depth would leave the net never having
  seen the shove/fold and deep-stack decisions that depend on it.
- **Advantage network** (`poker_ga/cfr_networks.py`): a plain MLP regressing
  each action category's counterfactual regret given the configured feature
  subset -- Single Deep CFR's only network, shared by every seat (no
  separate advantage net per player, no second strategy net).
- **Reservoir** (`poker_ga/cfr_reservoir.py`): a fixed-capacity uniform
  reservoir buffer (classic Algorithm R) of `(features, regrets,
  legal_mask, t)` samples collected during tree traversals. Sampling for
  insertion is uniform over every sample ever seen; the "later iterations
  matter more" weighting happens at training time instead, via each
  sample's stored `t` scaling its loss term.
- **Training loop** (`poker_ga/cfr_train.py`): alternates Monte Carlo tree
  traversals (filling the reservoir) with minibatch regression steps on the
  shared advantage network, weighting each sample's loss by its iteration
  number so later iterations dominate -- what gives Single Deep CFR its
  average-strategy approximation without a second policy network/reservoir.
  `Trainer` bundles the net/optimizer/reservoir a run advances;
  `run_iteration` runs one outer iteration's traversals + gradient steps.
- **Inference** (`poker_ga/cfr_policy.py`): wraps a trained `AdvantageNet`
  as a drop-in decision-maker for the real game engine, so a trained
  strategy can be handed straight to `simulate.run_session` /
  `benchmark.py` with zero changes to that tooling.
- **Benchmark** (`poker_ga/benchmark.py`): plays two pools of players
  head-to-head in 3-vs-3 tables, adding tables one at a time until a
  confidence interval around the bb/100 edge no longer straddles zero (or a
  hard table cap is hit) -- a statistically-resolved "did this actually
  help" check, used both by `cfr_main.py`'s training-loop checkpoint
  comparison and directly for one-off matches.
- **Session simulation** (`poker_ga/simulate.py`): plays a table through a
  session of hands, refilling any seat that busts with a fresh player
  (drawn from a backfill pool) rather than ending the session early --
  models a real cash table where play continues as players come and go.
- **Explorer** (`poker_ga/cfr_explorer.py`): an interactive Streamlit app
  for interrogating a trained checkpoint -- mark any feature as a filter, a
  group split, or a table split, and see the current net's average action
  distribution over the matching reservoir samples.

## Usage

Modules use flat imports and are run directly from inside `poker_ga/`
(not as a `-m` package):

```bash
pip install -r requirements.txt
cd poker_ga
python cfr_main.py --iterations 200 --traversals-per-iteration 200 --table-size 6
```

Key flags (see `python cfr_main.py --help` for all of them):

- `--iterations` / `--traversals-per-iteration` -- outer CFR iterations,
  and how many tree traversals (hands) feed the reservoir each iteration.
- `--sgd-steps-per-iteration` / `--batch-size` / `--lr` -- the regression
  step taken against the reservoir each iteration.
- `--reservoir-capacity` -- max samples kept in the uniform reservoir.
- `--num-equity-rollouts` -- board completions averaged for a pre-river
  all-in's equity estimate (exact whenever that many completions covers
  every possibility, Monte Carlo otherwise).
- `--min-starting-stack-bb` / `--max-starting-stack-bb` (default 20/200) --
  range each seat's own starting stack is independently redrawn from, per
  traversed hand (see CFR tree traversal above).
- `--hidden-sizes` / `--table-size` / `--feature-keys` -- advantage-net
  architecture and input vocabulary. Ignored (with a warning) when resuming
  from a checkpoint, since a saved net's shape can't change after the fact.
- `--starting-stack`, `--small-blind`, `--big-blind`,
  `--max-raises-per-street`, `--min-raise-fraction-of-pot` -- table rules
  for real (non-CFR) session play, e.g. the `--benchmark-interval` check --
  `--starting-stack` doesn't affect training traversals themselves, see
  `--min-starting-stack-bb` above.
- `--out-dir` / `--checkpoint-interval` / `--reload-previous` -- where
  checkpoints (`checkpoint_latest.{pt,json,npz}` + trainer state) are
  written, how often, and whether a run resumes from one automatically.
- `--benchmark-interval` (default 100) -- the progress check: every this
  many iterations, plays the current net head-to-head against a snapshot
  taken this many iterations ago, in 3-vs-3 tables, until the result is
  statistically resolved (`--benchmark-min/max-tables`,
  `--benchmark-table-batch`, `--benchmark-p-value`). The snapshot only
  advances to the current net on a resolved improvement, so a run of
  non-improving checks keeps comparing against the last net actually
  beaten. Set to 0 to disable.
- `--early-stop-patience` -- consecutive non-improving benchmark checks
  tolerated before training stops early (0 disables stopping; a
  non-improving check still reverts the net either way).
- `--workers` -- worker processes for the benchmark check specifically
  (tree traversal itself is sequential). 0 or negative uses every core.

The training loop's checkpoint is saved to `<out-dir>/checkpoint_latest`
(three files: `.pt` net weights, `.json` config, `.npz` reservoir, plus a
`_trainer_state.pt` sidecar with optimizer state and completed-iteration
count) after every `--checkpoint-interval` iterations, and reloaded
automatically on the next run against the same `--out-dir` unless
`--no-reload-previous` is passed. Load a trained policy back with:

```python
import cfr_policy
policy = cfr_policy.DeepCFRPolicy.from_checkpoint("cfr_runs/checkpoint_latest")
```

### Exploring a trained checkpoint

```bash
cd poker_ga
streamlit run cfr_explorer.py -- --checkpoint-path cfr_runs/checkpoint_latest
```

Mark any feature as a **Filter**, **Group split** (a separate table per
observed combination of values), or **Table split** (a row/column axis
within each table, capped at 2 features) in the sidebar, and see the
current net's average action distribution over the matching reservoir
samples -- action probabilities reflect the *current* net's regret-matching
strategy for each sampled situation, not the regret values originally
stored alongside it.

## Extending

- Add a feature: add a `FeatureSpec` to `FEATURE_SPECS` in `features.py`
  (a `group` from `FEATURE_GROUPS`, or a new one) and set its value in
  `extract_features`'s `values` dict, keyed by `spec.key`. It's
  automatically available to `--feature-keys` (or included by default,
  unless it's an `opp_`-prefixed opponent-tendency read).
- Add a new multi-value feature: give the parent spec a `value_table` (a
  tuple of `(normalized_value, human_label)`) and add one linked
  `FeatureSpec` per value with `kind="boolean"`, `linked_to=<parent key>`,
  and `linked_value_index=<index into value_table>`.
- Change the action space: `strategy.ACTION_CATEGORIES` and its sizing
  constants are the single source of truth, read by `cfr_actions.py`
  (translation to/from game actions), `cfr_networks.py` (the net's output
  dimension), and `cfr_explorer.py` (display).
- Pit two saved checkpoints (or a checkpoint against any other duck-typed
  policy) against each other by constructing `Player` objects and calling
  `simulate.run_session` or `benchmark.run_benchmark_until_resolved`
  directly.
