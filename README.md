# poker_ga

A genetic algorithm framework that evolves 6-max No-Limit Hold'em strategies.

## How it works

- **Genome** (`poker_ga/genome.py`): each feature gets *two* weights instead
  of one, feeding two near-orthogonal axes: **V** (showdown value — roughly
  "my equity against the range that continues") and **L** (leverage —
  roughly "how much of villain's range folds to me": fold equity shaped by
  blockers, initiative, board texture, position, SPR). Both are a linear
  sum of `weight x feature`, offset by a bias and clamped with plain min/max
  (not a curve) to land on 0-100 — read them as percentiles, e.g. `V=90` ~ a
  top-10% hand. They combine *non-convexly* (so it isn't just a 1D score
  again) into one action score: `A = max(V - theta_value, L - theta_bluff -
  kappa * V)`. Then: `A > 0` = bet/raise (sized at `(A / 100) x pot`), `elif
  V > theta_call` = call/check, `else` = fold/check. `theta_value` /
  `theta_bluff` / `theta_call` / `bias_v` / `bias_l` / `kappa` are all
  evolvable per-genome genes, initialized centered on the same reasonable
  values (`theta_value=theta_bluff=70`, `theta_call=40`, `bias_v=bias_l=50`,
  `kappa=0.5`) that were briefly used as fixed constants earlier in this
  project's life. They were fixed for a while because letting them
  free-drift under selection turned out to be exploitable — bet/raise fires
  on *either* axis clearing its bar (an OR) while folding needs *both* to
  fail (an AND), so selection could cheapen folding to near-zero by
  drifting any of these numbers toward the "makes it easier to clear a bar"
  side of either inequality (measured directly as a fold-rate collapse from
  ~34% to ~7% within a handful of generations). What actually turned out to
  be driving that collapse, though, was a game mechanic — sessions used to
  end outright once a couple of players busted, letting a lucky win get
  "locked in" before it could be punished — fixed by refilling busted seats
  with fresh players instead of ending the session (see Simulation below).
  With that root cause fixed, and with several other safeguards added since
  (a pot-scaled minimum raise floor, a sparsity penalty, island-isolated
  sub-populations, and benchmark-checkpoint reverting/early-stopping that
  catches and undoes a population that's measurably gotten worse — see
  Island model and Resuming and benchmark checkpoints below), these six
  scalars are evolvable again: the mechanism that made an unbounded drift
  pay off is gone, and there's now a safety net that reverts training if a
  similar drift ever did start paying off. Weights live on the same 0-100
  range as V/L (sized so a handful of active features move V/L
  meaningfully) and biases are centered at 50 — the "no information"
  percentile — so every number in a genome reads the same way a human would
  think about it — no unit conversion needed at the table, and the only
  non-arithmetic step anywhere is the min/max clamp. `NUM_FEATURES` drives
  every weight vector's shape automatically, whatever `features.py`
  defines. The feature weights themselves (`weights_v`/`weights_l`) are
  quantized to a small fixed alphabet — `WEIGHT_ALPHABET = {-30, -20, -10,
  -5, 0, 5, 10, 20, 30}` — via `quantize()`, applied inside `mutate_weights()`
  and `Genome.random()` so every weight gene snaps back onto the alphabet
  without `ga.py` needing to know which genes are weights. This turns a
  genome into something closer to a lookup table a human could memorize,
  rather than an arbitrary-float vector.
- **Ranges** (`poker_ga/ranges.py`): parses human-readable starting-hand
  range strings, e.g. `"AA-77, AJs+, AQo+, KQs"`, into the standard 169-hand
  abstraction (13 pocket pairs + 78 suited + 78 offsuit two-card combos,
  ignoring exact suit — preflop, all 4 suits of a suited combo play
  identically, and likewise for offsuit combos, so this is the same
  abstraction every range tool/chart uses). `parse_range()` supports exact
  hands (`"77"`, `"AKs"`, `"AKo"`), pair ranges/`"+"` (`"AA-77"`, `"77+"`),
  suited/offsuit `"+"` with a fixed top card (`"AJs+"` = AJs, AQs, AKs),
  fixed-top-card ranges (`"AJs-A5s"`), and matching-gap connector ranges
  (`"T9s-54s"` = T9s, 98s, 87s, 76s, 65s, 54s). `hand_label()` computes a
  hole-card pair's canonical label (e.g. `"AKs"`) so it can be checked
  against a parsed range.
- **GTO spots** (`poker_ga/gto.py`): the piece that lets a genome memorize
  an *exact* strategy for a specific, well-defined spot — the way a human
  plays "UTG open, 100BB" straight off a memorized chart rather than by
  feel — instead of only ever reasoning through the linear V/L system. A
  `GTOSpot` combines a `SpotMatcher` (a readable, declarative definition of
  which situations it applies to: street, pot type, position, facing a bet
  or not, who was last aggressor, effective stack in actual big blinds) with
  `action_ranges` — an ordered list of `(action, range_str)` pairs (e.g.
  `("raise_2.5bb", "77+, ATs+, ...")`, `("call", "22-99, ...")`) checked in
  order, first match wins; anything not covered by any listed range falls
  back to `default_action` (normally `"fold"`, matching how a real chart is
  read — everything not colored in is a fold). Action tokens are `"fold"`,
  `"call"`, `"raise_NN"` (NN = percent of pot, e.g. `"raise_75"` = 3/4-pot;
  the raise amount is an increment on top of whatever's already bet, same
  as the linear V/L system's sizing), `"raise_NNbb"` (raise so this
  player's *total* commitment this street reaches NN big blinds, e.g.
  `"raise_2.5bb"` — the natural way to write a preflop open size; exact for
  any seat that hasn't put any chips in yet this street, a close
  approximation for the blinds), or `"allin"` (shoves the full stack).
  `GTO_SPOTS` is a small, fixed,
  code-defined catalog (same pattern as `features.py`'s `FEATURE_SPECS` —
  extend it by adding entries, not by making it runtime-configurable); the
  included spots are illustrative, reasonable ranges, not verified solver
  output, so swap in real solved charts for genuine accuracy. Note that
  `SpotMatcher` can express "BTN facing a 3-bet" but not "BTN facing a
  3-bet specifically from the BB" — `Situation` tracks whether *I* was the
  last aggressor, not which seat/position made a given raise. Position
  matching reuses the same `seating.seat_role()` helper as the rest of the
  framework, so a spot's definition of "BTN" means exactly what the
  Starting Seat Position feature means, and stack depth is measured in
  actual big blinds (`Situation.big_blind`, threaded from `game.py`), so
  "100BB stack" means what it says regardless of `--starting-stack`.
  Whether a genome actually *uses* a given spot is a separate, evolvable
  per-genome gene (`gto_flags` in `genome.py`, one boolean per `GTO_SPOTS`
  entry, initialized on ~10% of the time — see `GTO_INIT_PROB`) — a spot
  existing in the catalog doesn't mean every genome plays it that way, only
  that a genome *can* learn to trust it. When a genome has a spot active and
  the current situation matches it, `Genome.decide()` looks the hand up in
  that spot's chart and plays exactly what it says, **bypassing V/L
  entirely** for that decision (checked in `GTO_SPOTS` catalog order; the
  first active, matching spot wins) — this is deliberately a hard override
  rather than another linear term, since a chart lookup is exactly as
  memorizable for a human as the chart itself, whereas folding "always raise
  AA in this one exact spot" into the linear weights would require it to
  also make sense as a general trend across every other spot, which it
  doesn't. `gto_flags` get their own mutation (bit-flip) and crossover
  (uniform, like the quantized weights) operators, and don't count toward
  `nonzero_weight_count()` (a different kind of complexity — memorized
  charts, not linear weights). Exported strategy reports include a "GTO
  Chart Overrides" section listing every active spot's full chart (see
  Final tournament output below).
- **Features** (`poker_ga/features.py`): 16 basic situation characteristics
  (made-hand category, hole/shared-cards high card, hole card connectivity,
  street, call size vs pot, SPR, action-order position, starting seat
  position, stack depth, players in hand, raises this street, pot type, and
  3 flop-texture dimensions) each get *two* representations: one generalized
  0-1 feature so the genome can learn a linear trend across its values, and
  one exact indicator feature per specific value (e.g. one boolean each for
  Preflop/Flop/Turn/River, one for every card rank 2-Ace, one for each of
  UTG/HJ/CO/BTN/SB/BB, one for Rainbow/Flush-Draw-Flop/Monotone, one for
  Unraised/Single-Raised/3-Bet/4-Bet+ pots) so the genome can also learn a
  non-linear, value-specific override. Plus ~20 standalone 0/1 heuristic
  flags that don't reduce to a clean one-hot family (top pair, overpair,
  combo draw, backdoor flush draws, connected flop, etc — some, like Low
  Pair, are subsets of others, like Underpair, so forcing them into a
  single mutually-exclusive category would be wrong). That's 130 features
  total, each with a human-readable label, a precise definition, and a
  high-level group (e.g. "Made Hand Features", "Draw Features" — see
  `FEATURE_GROUPS`) used to organize the strategy exports below (see
  `FEATURE_SPECS`). There's no separate Pot Odds feature — it's the same
  information as Call Size Vs Pot, just a harder mental calculation from
  that ratio, so only the easier-to-use one is kept. Pot Type (Unraised /
  Single Raised / 3-Bet / 4-Bet+) is frozen at the final preflop raise count
  once the flop is dealt, unlike Raises This Street, which resets every
  street — "we're in a 3-bet pot" describes the whole hand.
- **Seating** (`poker_ga/seating.py`): pure seat-arithmetic shared by the
  game engine and feature extraction — blind positions, preflop action
  order, and standard seat-role naming (UTG/HJ/CO/BTN/SB/BB) that correctly
  collapses at smaller table sizes (e.g. 4-handed only has UTG and BTN as
  non-blind seats; heads-up the button is the small blind).
- **Game engine** (`poker_ga/game.py`): a real 6-max NLHE implementation —
  blinds, four streets, all-in handling with correct multi-way side pots,
  and showdown.
- **Simulation** (`poker_ga/simulate.py`): each generation, the population is
  repeatedly reshuffled into random 6-max tables ("rounds"). Each table plays
  a session of hands (starting stacks reset per session) until a hand cap is
  hit. Any seat whose stack hits zero is immediately **refilled** with a
  fresh player (a full starting stack) drawn from the whole population,
  rather than ending the session early -- models a real cash table, where
  the game keeps going as players bust and new ones sit down. (An earlier
  design ended a table's session outright once a couple of its original
  players had busted, to avoid over-adapting to short-handed end-games --
  but that let a genome "lock in" a lucky win by causing the session to end
  before it could be punished for the risk, which turned out to be the
  dominant driver of a fold-rate collapse across generations: fixing every
  other suspect gene didn't stop it, but disabling early termination did.
  Refilling gets the short-handed-avoidance benefit without that exploit,
  since the table stays near-full for the whole session.) A player's fitness
  is their net chip result (across however many stints they play, original
  seating or backfill) summed across all sessions that generation.
  `run_generation` also returns a `GenerationStats` summary (mean hands
  survived per stint, bust rate, fold rate when facing a bet, mean raises
  per street) printed every generation as a sense check against the same
  kind of collapse recurring.
- **GA** (`poker_ga/ga.py`): tournament selection with elitism carrying the
  top genomes forward unchanged. Both crossover and mutation
  (`Genome.crossover` / `Genome.mutate` in `genome.py`) are gene-type-aware:
  `noise_std` (the only remaining continuous scalar gene -- see Genome
  above) gets ordinary blend crossover and additive-gaussian mutation, but
  the quantized feature weights get dedicated discrete operators instead,
  since blending/adding noise to a value from a small fixed alphabet and
  re-quantizing behaves badly in two different ways. Crossover: blending an
  exact alphabet value from each parent and rounding invents intermediate
  values neither parent had, and systematically dilutes sparsity -- a 0
  gene crossed with a large nonzero gene mostly produces a nonzero child,
  since only the narrow slice of the blend range nearest 0 rounds back to 0
  (measured: a fully-sparse parent crossed with a fully-dense one kept only
  ~26% of its zeros under blending). So weight genes instead use uniform
  (discrete) crossover: each gene is inherited whole from one parent or the
  other (50/50), which preserves sparsity exactly on average and never
  invents values. Mutation: additive noise re-quantized would almost always
  land back on the same WEIGHT_ALPHABET value it started at, since the
  alphabet's ~5-10-unit gaps dwarf any sane continuous mutation step,
  silently making weight mutation a no-op. Each selected weight gene
  instead gets one of: a one-step nudge up/down the alphabet (local
  search), or a jump to a uniformly random alphabet value (occasional
  larger jumps, including landing back on 0).
- **Island model** (`IslandConfig` / `IslandModel` in `poker_ga/ga.py`):
  `--population` is split into `--num-islands` (default 3) independent
  `Population`s, each with its own breeding pool *and* its own tables --
  `main.py` calls `run_generation` once per island, so an island's players
  only ever face other members of that same island, never the other
  islands. That isolation is the point: if one island's fitness landscape
  gets taken over by a pathological strategy (the "never fold" collapse
  this project spent a while diagnosing is exactly this kind of failure),
  the other islands aren't directly exposed to it as opponents, so they
  don't automatically inherit the same collapse -- diversity across the
  whole population survives even if one island doesn't. Every
  `--migration-interval` generations (default 10), a ring migration is the
  only channel connecting islands: each island's best `--migration-size`
  (default 3) genomes this generation are copied into the next island in
  the ring, overwriting random *non-elite* slots there (elitism already
  protects each island's own best; "worst" isn't used as the criterion
  since the receiving island's freshly-bred next generation hasn't been
  evaluated yet at the point migration happens, and fitness numbers aren't
  directly comparable across islands anyway -- different opponents,
  different noise). `player_id`s are offset per island (1,000,000 apart) so
  they never collide once islands are combined for the final tournament,
  benchmark checkpoints, or `--reload-previous`. Set `--num-islands 1` to
  disable and get the original single-population behavior exactly (no
  migration, no per-island output line).
- **Final tournament** (`poker_ga/tournament.py`): once evolution finishes,
  the last generation is re-evaluated with many more rounds and a higher
  hand cap than the (deliberately cheap) per-generation fitness pass, for a
  low-variance final ranking. The top N genomes are exported as
  human-readable strategy reports plus performance stats.

## Usage

Modules use flat imports and are run directly from inside `poker_ga/`
(not as a `-m` package):

```bash
pip install -r requirements.txt
cd poker_ga
python main.py --generations 50 --population 90 --rounds 3
```

Key flags (see `python main.py --help` for all of them):

- `--population` — total pool size across all islands combined, must be a
  multiple of `--num-islands * 6`.
- `--num-islands` (default 3), `--migration-interval` (default 10),
  `--migration-size` (default 3) — the island model (see Island model
  above). Set `--num-islands 1` to disable and evolve a single population,
  as before.
- `--rounds` — random table re-seatings per generation (more rounds = less
  variance in the fitness signal, at the cost of speed).
- `--max-hands` — hand cap per table session during evolution (a session
  runs the full hand cap regardless of busts — see Simulation above —
  since busted seats are refilled rather than ending the session).
- `--starting-stack`, `--small-blind`, `--big-blind` — table stakes.
- `--elite`, `--mutation-rate`, `--mutation-scale` — GA hyperparameters.
- `--sparsity-penalty` (default 2.0) — chips subtracted from fitness per
  nonzero feature weight (`weights_v` + `weights_l` combined, out of a
  possible `2 x NUM_FEATURES`), applied both during evolution and to the
  final tournament ranking. Pushes selection toward genomes where most
  weights land on exactly 0 — a shorter, more memorizable "cheat sheet" —
  alongside raw chip performance. Set to 0 to disable.
- `--benchmark-interval` (default 10) — every this many generations, plays
  the current population (combined across all islands) head-to-head against
  a saved checkpoint from `--benchmark-interval` generations ago, in
  `--benchmark-tables` (default 2400) independent 3-vs-3 tables, and prints
  aggregate net chips + bb/100 for each side. Unlike the per-generation
  fitness number (only comparable against that generation's own random
  opponents, not across generations), this is a direct, apples-to-apples
  measure of whether evolution is actually improving. Set to 0 to disable.
  See Benchmark checkpoints below.
- `--final-rounds`, `--final-max-hands` — size of the final scoring
  tournament run after evolution completes (bigger = lower-variance ranking,
  slower). Defaults (200 rounds, 100-hand cap) take roughly 1-2 minutes at
  the default population of 180.
- `--top-n` — how many top genomes to export.
- `--final-out-dir` — where reports go (defaults to `<out-dir>/final`).
- `--reload-previous` (default on) — seeds generation 0 from a previous
  run's saved population instead of starting from scratch, so consecutive
  runs against the same `--out-dir` keep evolving where the last one left
  off. Prefers `<out-dir>/latest_population.json` (updated every generation
  -- see below) if present, falling back to `<final-out-dir>/population.json`
  (only written once, when a run finishes its final tournament) otherwise.
  If population sizes differ, the reloaded genomes (best-first) are
  truncated or padded with fresh random genomes to fit. Pass
  `--no-reload-previous` to always start random, or `--reload-path` to
  reload from a specific file.
- `--early-stop-patience` (default 3) — see Benchmark checkpoints below;
  how many consecutive non-improving benchmark checks are tolerated before
  training stops early. 0 disables stopping (reverting on non-improvement
  still happens, training just never gives up).

The best genome is saved after every generation to `<out-dir>/best_genome_latest.json`.
Load it back with:

```python
from genome import Genome
best = Genome.load("runs/best_genome_latest.json")
```

Genomes are saved as a named JSON dictionary (feature/GTO-spot key -> value),
not a raw positional array, so a saved genome survives `features.py`/`gto.py`
changing in a later version of the code: `Genome.load`/`load_population`
drop any saved entry whose name no longer exists (printing a warning) and
freshly randomize any entry the current catalog expects but the save doesn't
have (also with a warning), rather than erroring out or silently
misaligning weights to the wrong features.

### Resuming and benchmark checkpoints

Two population snapshots are kept on disk, both **continuously overwritten
in place** (never one file per generation, so long runs don't bloat disk):

- `<out-dir>/latest_population.json` — the full population, saved after
  *every* generation. This is what `--reload-previous` prefers, so an
  interrupted or killed run can resume from wherever it last got to, not
  just from a fully completed run's final tournament output.
- `<out-dir>/benchmarks/checkpoint_population.json` — a full population
  snapshot, but only ever advanced when a benchmark check (below) confirms
  the current population actually beat it. It always holds the last
  population that was *measured* to be an improvement, not just the most
  recent one.

Every `--benchmark-interval` generations (default 10, 0 disables), the live
population is played head-to-head against `checkpoint_population.json` in
`--benchmark-tables` independent 3-vs-3 tables (`benchmark.run_benchmark`;
3 random players from each side per table, refilling any busted seat with a
fresh player from its own side so the match stays 3v3 for the whole
session), printing a line like:

```
         | benchmark vs checkpoint | current    -941.4 chips ( -104.60 bb/100) | checkpoint    +941.4 chips ( +104.60 bb/100) | NOT IMPROVED
```

This is the tangible cross-generation signal the per-generation fitness
number can't give you (that number only ranks a generation's genomes
against each other's own random opponents that generation, so "500" at gen
10 and "500" at gen 50 mean nothing relative to each other). The match is
zero-sum (chips only move between the seated players, refilling never
creates or destroys any), so "improved" is exactly `current_net_total > 0`:

- **Improved** — `checkpoint_population.json` is overwritten with the
  current population (the checkpoint advances), and the consecutive
  non-improvement counter resets to 0.
- **Not improved** — training reverts: the live population is discarded and
  replaced with the checkpoint's (both `latest_population.json` and the
  in-memory population), so the next generation retries evolution from the
  same starting point with fresh randomness rather than building further on
  a population that got measurably worse. The consecutive non-improvement
  counter increments; if it reaches `--early-stop-patience` (default 3),
  training stops early and jumps straight to the final tournament instead
  of running out the rest of `--generations`.

### Final tournament output

After the last generation, `<out-dir>/final/` contains:

- `leaderboard.md` — a ranked table (mean net chips/session, win rate, bust
  rate, bb/100, nonzero weight count) for the top N genomes.
- `rankNN_playerID_strategy.md` — one report per top genome: performance
  stats, this genome's own theta_value/theta_bluff/theta_call/bias_v/
  bias_l/kappa/noise_std (see Genome above), its nonzero feature weight
  count (out of `2 x NUM_FEATURES` possible), a `## GTO Chart Overrides`
  section (see GTO spots above) listing every spot this genome currently
  trusts — its matcher description and its full action/range table, exactly
  as a human would need to memorize it — then a `##
  Feature Groups` section organized by theme (Hole Card Characteristics,
  Board / Flop Characteristics, Made Hand Features, Draw Features, Betting
  Behaviour Features, Stack & Pot Features, Table & Game State Features —
  see `FEATURE_GROUPS`). Each group shows its standalone boolean (0/1)
  features as a compact table (V weight, L weight), then a per-value
  breakdown table for each of its multi-value features (e.g. Betting Street
  broken into Preflop/Flop/Turn/River, each row showing its V and L
  weight). Each breakdown row combines that value's general (linear) weight
  with its own exact indicator feature's weight into one raw (pre-clip) number
  per axis, so the ~90 underlying indicator features never clutter the
  report as separate entries — these weights are on the raw pre-clip scale
  (and quantized, per WEIGHT_ALPHABET), not literal V/L percentage points. A
  reference section defines each generalized/standalone feature precisely,
  grouped the same way (36 entries, not 130).
- `rankNN_playerID_genome.json` — the named-dictionary weights, loadable via `Genome.load`.
- `population.json` — the entire final generation, ranked best-first, saved
  via `genome.save_population`. This is what `--reload-previous` picks up
  on the next run.

## Extending

- Add a standalone boolean feature: add a `FeatureSpec` to `FEATURE_SPECS` in
  `features.py` with a `group` (one of `FEATURE_GROUPS`, or a new one you
  add to that list), and set its value in `extract_features`'s `values`
  dict (keyed by `spec.key`, so ordering doesn't matter) — genomes
  auto-resize since `NUM_FEATURES` drives every weight shape.
- Add a new multi-value feature: give the parent spec a `value_table` (a
  tuple of `(normalized_value, human_label)`) and a `group`, then add one
  linked `FeatureSpec` per value with `kind="boolean"`,
  `linked_to=<parent key>`, and `linked_value_index=<index into
  value_table>` — `_linked_bool` / `_continuous_children` in `features.py`
  show the pattern. Linked children don't need their own `group`; they
  inherit the parent's via `group_of()`. Linking is also what makes the
  strategy export fold each value into one combined row instead of listing
  every indicator separately.
- Change fitness: edit `run_session`/`run_generation` in `simulate.py` — e.g.
  blend in hands-survived, or weight later generations' sessions differently.
- Pit two saved genomes against each other head-to-head by constructing
  `Player` objects around loaded genomes and calling `simulate.run_session`
  directly (also used internally by `tournament.run_final_tournament`).
