# poker_ga

A genetic algorithm framework that evolves 6-max No-Limit Hold'em strategies.

## How it works

- **Genome** (`poker_ga/genome.py`): each player is one linear scoring
  function, simple enough to hand-compute — `score = bias + sum(weight x
  feature)` over ~26 features, thresholded into an action: `score <= 0` =
  fold (check if nothing to call), `0 < score <= 1` = check/call, `score >
  1` = bet/raise sized at `(score - 1) x pot` (e.g. 2.0 = a pot-sized raise).
  A player is ~28 numbers total (one weight per feature, a bias, and a
  noise-stddev for bluffing/exploration) — those numbers *are* the genes the
  GA evolves.
- **Features** (`poker_ga/features.py`): ~26 basic situation characteristics
  fed into that scoring function — made-hand category, explicit flags for
  pair/straight/flush/etc., high card, flush/straight draws, hole card
  texture, street, pot odds, SPR, position, whether I'm facing a bet, whether
  I was the last aggressor, and more. Each has a human-readable label and a
  precise definition (see `FEATURE_SPECS`), which is what shows up in
  exported strategy reports.
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
  stats, its bias and noise level, every feature's weight (sorted by
  magnitude), and a reference section defining each feature precisely.
- `rankNN_playerID_genome.npy` — the raw weights, loadable via `Genome.load`.
- `population.npy` — the entire final generation, ranked best-first, saved
  via `genome.save_population`. This is what `--reload-previous` picks up
  on the next run.

## Extending

- Add a feature: append to `FEATURE_NAMES`/`extract_features` in `features.py`
  (genomes auto-resize since `NUM_FEATURES` drives every weight shape).
- Change fitness: edit `run_session`/`run_generation` in `simulate.py` — e.g.
  blend in hands-survived, or weight later generations' sessions differently.
- Pit two saved genomes against each other head-to-head by constructing
  `Player` objects around loaded genomes and calling
  `tournament.run_session_detailed` directly.
