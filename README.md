# poker_ga

A genetic algorithm framework that evolves 6-max No-Limit Hold'em strategies.

## How it works

- **Genome** (`poker_ga/genome.py`): each feature gets *two* weights instead
  of one, feeding two near-orthogonal axes: **V** (showdown value — roughly
  "my equity against the range that continues") and **L** (leverage —
  roughly "how much of villain's range folds to me": fold equity shaped by
  blockers, initiative, board texture, position, SPR). Both are a linear
  sum of `weight x feature`, offset by a fixed bias and clamped with plain
  min/max (not a curve) to land on 0-100 — read them as percentiles, e.g.
  `V=90` ~ a top-10% hand. They combine *non-convexly* (so it isn't just a
  1D score again) into one action score: `A = max(V - THETA_VALUE, L -
  THETA_BLUFF - KAPPA * V)`. Then: `A > 0` = bet/raise (sized at `(A / 100)
  x pot`), `elif V > THETA_CALL` = call/check, `else` = fold/check.
  `THETA_VALUE`/`THETA_BLUFF`/`THETA_CALL`/`BIAS_V`/`BIAS_L`/`KAPPA` are all
  **fixed module-level constants, not evolvable genes** — they started out
  as per-genome genes, but that let each one random-walk without any bound.
  Since bet/raise fires on *either* axis clearing its bar (an OR) while
  folding needs *both* to fail (an AND), selection could cheapen folding to
  near-zero by drifting *any* number that sits on the "makes it easier to
  clear a bar" side of either inequality — measured directly: fixing just
  the thresholds didn't stop a fold-rate collapse (~34% to ~7% within a
  handful of generations, average hands survived per session collapsing to
  ~1); evolution simply moved the same exploit onto `bias_l` (drifted
  toward the value that pushes `L` past `THETA_BLUFF` on its own) and
  `kappa` (drifted toward zero, disabling the term that's supposed to make
  bluffing harder as `V` rises). There's no real reason any of these need
  to be learned per genome — a human would pick one sensible set of numbers
  and stick with it, same as here. Only the feature weights and the
  exploration noise (`noise_std`) are left to evolve. Weights live on the
  same 0-100 range as V/L (sized so a handful of active features move V/L
  meaningfully) and `BIAS_V`/`BIAS_L` sit at 50 — the "no information"
  percentile — so every number in a genome reads the same way a human would
  think about it — no unit conversion needed at the table, and the only
  non-arithmetic step anywhere is the min/max clamp. `NUM_FEATURES` drives
  every weight vector's shape automatically, whatever `features.py`
  defines. The feature weights themselves (`weights_v`/`weights_l`) are
  quantized to a small fixed alphabet — `WEIGHT_ALPHABET = {-30, -20, -10,
  -5, 0, 5, 10, 20, 30}` — via `quantize()`, applied centrally in
  `Genome.unflatten()` so every genome coming out of crossover/mutation
  snaps back onto the alphabet without `ga.py` needing to know which genes
  are weights. This turns a genome into something closer to a lookup table
  a human could memorize, rather than an arbitrary-float vector.
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
- `--reload-previous` (default on) — seeds generation 0 from the previous
  run's saved final population (`<final-out-dir>/population.npy`) instead of
  starting from scratch, so consecutive runs against the same `--out-dir`
  keep evolving where the last one left off. If population sizes differ,
  the reloaded genomes (best-first) are truncated or padded with fresh
  random genomes to fit. Pass `--no-reload-previous` to always start random,
  or `--reload-path` to reload from a specific file.

The best genome is saved after every generation to `<out-dir>/best_genome_latest.npy`.
Load it back with:

```python
from genome import Genome
best = Genome.load("runs/best_genome_latest.npy")
```

### Benchmark checkpoints

Every `--benchmark-interval` generations (default 10, 0 disables), `<out-
dir>/benchmarks/gen{N:05d}_population.npy` is saved — a full snapshot of
that generation's population (`genome.save_population`). Once a checkpoint
exists from `--benchmark-interval` generations before the current one,
`benchmark.run_benchmark` seats it against the live population in
`--benchmark-tables` independent 3-vs-3 tables (3 random players from each
side per table, refilling any busted seat with a fresh player from its own
side so the match stays 3v3 for the whole session), and prints a line like:

```
         | benchmark vs gen    0 | current    -941.4 chips ( -104.60 bb/100) | checkpoint    +941.4 chips ( +104.60 bb/100)
```

This is the tangible cross-generation signal the per-generation fitness
number can't give you (that number only ranks a generation's genomes
against each other's own random opponents that generation, so "500" at gen
10 and "500" at gen 50 mean nothing relative to each other) — a positive
`current` bb/100 means the population has genuinely gotten better since the
compared checkpoint, not just adapted to whatever it was facing this
generation.

### Final tournament output

After the last generation, `<out-dir>/final/` contains:

- `leaderboard.md` — a ranked table (mean net chips/session, win rate, bust
  rate, bb/100, nonzero weight count) for the top N genomes.
- `rankNN_playerID_strategy.md` — one report per top genome: performance
  stats, the fixed THETA_VALUE/THETA_BLUFF/THETA_CALL/BIAS_V/BIAS_L/KAPPA
  constants (same for every genome — see Genome above) plus this genome's
  own noise_std, its nonzero feature weight count (out of `2 x
  NUM_FEATURES` possible), then a `##
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
- `rankNN_playerID_genome.npy` — the raw weights, loadable via `Genome.load`.
- `population.npy` — the entire final generation, ranked best-first, saved
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
