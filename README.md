# poker_ga

A genetic algorithm framework that evolves 6-max No-Limit Hold'em strategies.

## How it works

- **Genome** (`poker_ga/genome.py`): each feature gets *two* weights instead
  of one, feeding two near-orthogonal axes: **V** (showdown value — roughly
  "my equity against the range that continues") and **L** (leverage —
  roughly "how much of villain's range folds to me": fold equity shaped by
  blockers, initiative, board texture, position, SPR). Both are a linear
  sum of `weight x feature`, offset and clipped to read as a 0-100
  percentage (no exponentials, so the mapping can be done by hand). They
  combine *non-convexly* (so it isn't just a 1D score again) into one
  action score: `A = max(V - theta_value, L - theta_bluff - kappa * V)`.
  Then: `A > 0` = bet/raise (sized at `(A / 100) x pot`), `elif
  V > theta_call` = call/check, `else` = fold/check. theta_value/theta_bluff
  /theta_call are stored as raw genes and linearly rescaled onto the same
  0-100 scale only when used, so every gene — weights, biases, thresholds,
  kappa, noise — mutates at a comparable numeric scale. `NUM_FEATURES`
  drives every weight vector's shape automatically, whatever `features.py`
  defines. The feature weights themselves (`weights_v`/`weights_l`, not the
  biases/thresholds) are quantized to a small fixed alphabet —
  `WEIGHT_ALPHABET = {-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3}` — via `quantize()`,
  applied centrally in `Genome.unflatten()` so every genome coming out of
  crossover/mutation snaps back onto the alphabet without `ga.py` needing to
  know which genes are weights. This turns a genome into something closer
  to a lookup table a human could memorize, rather than an arbitrary-float
  vector.
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
  a session of hands (starting stacks reset per session) until only one
  player has chips left or a hand cap is hit; players are removed from the
  table the instant they bust. A player's fitness is their net chip result
  summed across all sessions they played that generation.
- **GA** (`poker_ga/ga.py`): tournament selection + blend crossover + gaussian
  mutation, with elitism carrying the top genomes forward unchanged.
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
python main.py --generations 50 --population 60 --rounds 3
```

Key flags (see `python main.py --help` for all of them):

- `--population` — pool size, must be a multiple of 6.
- `--rounds` — random table re-seatings per generation (more rounds = less
  variance in the fitness signal, at the cost of speed).
- `--max-hands` — hand cap per table session during evolution.
- `--busts-before-table-ends` (default 2) — a table's session ends once this
  many of its original players have busted, instead of always playing down
  to heads-up. Models real tables refilling empty seats with new players,
  and keeps the GA from over-adapting to short-handed end-games (where
  optimal play looks very different) that aren't representative of a normal
  full/near-full table. Applies to both evolution and the final tournament.
- `--starting-stack`, `--small-blind`, `--big-blind` — table stakes.
- `--elite`, `--mutation-rate`, `--mutation-scale` — GA hyperparameters.
- `--sparsity-penalty` (default 2.0) — chips subtracted from fitness per
  nonzero feature weight (`weights_v` + `weights_l` combined, out of a
  possible `2 x NUM_FEATURES`), applied both during evolution and to the
  final tournament ranking. Pushes selection toward genomes where most
  weights land on exactly 0 — a shorter, more memorizable "cheat sheet" —
  alongside raw chip performance. Set to 0 to disable.
- `--final-rounds`, `--final-max-hands` — size of the final scoring
  tournament run after evolution completes (bigger = lower-variance ranking,
  slower). Defaults (200 rounds, 500-hand cap) take roughly 1-2 minutes at
  the default population of 96.
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

### Final tournament output

After the last generation, `<out-dir>/final/` contains:

- `leaderboard.md` — a ranked table (mean net chips/session, win rate, bust
  rate, bb/100, nonzero weight count) for the top N genomes.
- `rankNN_playerID_strategy.md` — one report per top genome: performance
  stats, its theta_value/theta_bluff/theta_call/kappa/noise_std, its nonzero
  feature weight count (out of `2 x NUM_FEATURES` possible), then a `##
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
  `Player` objects around loaded genomes and calling
  `tournament.run_session_detailed` directly.
