# poker_ga

A genetic algorithm framework that evolves 6-max No-Limit Hold'em strategies.

## How it works

- **Genome** (`poker_ga/genome.py`): each player is one linear scoring
  function, simple enough to hand-compute — `score = bias + sum(weight x
  feature)`, thresholded into an action: `score <= 0` = fold (check if
  nothing to call), `0 < score <= 1` = check/call, `score > 1` = bet/raise
  sized at `(score - 1) x pot` (e.g. 2.0 = a pot-sized raise). Those weights
  *are* the genes the GA evolves; `NUM_FEATURES` drives their shape
  automatically, whatever `features.py` defines.
- **Features** (`poker_ga/features.py`): 12 basic situation characteristics
  (made-hand category, high card, hole card connectivity, street, pot odds,
  SPR, action-order position, starting seat position, stack depth, players
  in hand, raises this street) each get *two* representations: one
  generalized 0-1 feature so the genome can learn a linear trend across its
  values, and one exact indicator feature per specific value (e.g. one
  boolean each for Preflop/Flop/Turn/River, one for every card rank 2-Ace,
  one for each of UTG/HJ/CO/BTN/SB/BB) so the genome can also learn a
  non-linear, value-specific override. Plus a handful of standalone 0/1
  flags (facing a bet, suited hole cards, etc). That's 97 features total,
  each with a human-readable label and a precise definition (see
  `FEATURE_SPECS`).
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
- `--starting-stack`, `--small-blind`, `--big-blind` — table stakes.
- `--elite`, `--mutation-rate`, `--mutation-scale` — GA hyperparameters.
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
  rate, bb/100) for the top N genomes.
- `rankNN_playerID_strategy.md` — one report per top genome: performance
  stats, its bias and noise level, a weight table for the standalone boolean
  (0/1) features, and a per-value breakdown table for every multi-value
  feature (e.g. Betting Street broken into Preflop/Flop/Turn/River, High
  Card Rank into 2 through Ace, Pot Odds into named risk ratios). Each
  breakdown row combines that value's general (linear) contribution with its
  own exact indicator feature's contribution into one number, so the ~80
  underlying indicator features never clutter the report as separate
  entries. A reference section defines each generalized/standalone feature
  precisely (17 entries, not 91).
- `rankNN_playerID_genome.npy` — the raw weights, loadable via `Genome.load`.
- `population.npy` — the entire final generation, ranked best-first, saved
  via `genome.save_population`. This is what `--reload-previous` picks up
  on the next run.

## Extending

- Add a standalone boolean feature: add a `FeatureSpec` to `FEATURE_SPECS` in
  `features.py` and set its value in `extract_features`'s `values` dict
  (keyed by `spec.key`, so ordering doesn't matter) — genomes auto-resize
  since `NUM_FEATURES` drives every weight shape.
- Add a new multi-value feature: give it a `value_table` (a tuple of
  `(normalized_value, human_label)`), then add one linked `FeatureSpec` per
  value with `kind="boolean"`, `linked_to=<parent key>`, and
  `linked_value_index=<index into value_table>` — `_linked_bool` /
  `_continuous_children` in `features.py` show the pattern. Linking is what
  makes the strategy export fold each value into one combined row instead of
  listing every indicator separately.
- Change fitness: edit `run_session`/`run_generation` in `simulate.py` — e.g.
  blend in hands-survived, or weight later generations' sessions differently.
- Pit two saved genomes against each other head-to-head by constructing
  `Player` objects around loaded genomes and calling
  `tournament.run_session_detailed` directly.
