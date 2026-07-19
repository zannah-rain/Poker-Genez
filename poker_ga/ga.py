"""Genetic algorithm operators: selection, crossover, mutation, and the
population container that turns a generation's fitness scores into the
next generation of genomes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from genome import Genome
from player import Player


@dataclass
class GAConfig:
    population_size: int = 60  # keep a multiple of 6 for clean table seating
    elite_count: int = 4  # top genomes copied unchanged into the next generation
    tournament_size: int = 4  # contestants per selection tournament
    crossover_rate: float = 0.7  # probability of blending two parents vs. cloning one
    mutation_rate: float = 0.15  # per-gene probability of mutation
    mutation_scale: float = 0.3  # stddev of mutation noise
    init_scale: float = 0.5  # stddev of initial random genes


def tournament_select(players: list[Player], fitness: dict[int, float], k: int, rng: np.random.Generator) -> Player:
    contestants = rng.choice(len(players), size=min(k, len(players)), replace=False)
    best = max(contestants, key=lambda idx: fitness[players[idx].player_id])
    return players[best]


def crossover(a: Genome, b: Genome, rng: np.random.Generator) -> Genome:
    """Delegates to Genome.crossover, which knows to use uniform (discrete)
    inheritance for the quantized feature weights and blend crossover for
    the continuous scalars -- see genome.py."""
    return a.crossover(b, rng)


def mutate(genome: Genome, rate: float, scale: float, rng: np.random.Generator) -> Genome:
    """Delegates to Genome.mutate, which knows to mutate the quantized
    feature weights differently (alphabet jumps) from the continuous
    scalars (additive gaussian, using `scale`) -- see genome.py."""
    return genome.mutate(rng, rate, scale)


class Population:
    def __init__(
        self,
        config: GAConfig,
        rng: np.random.Generator,
        seed_genomes: list[Genome] | None = None,
        id_offset: int = 0,
    ):
        """`seed_genomes`, if given, seeds generation 0 (e.g. reloaded from a
        previous run's final population) instead of starting fully random.
        Best-first ordering matters if len(seed_genomes) > population_size,
        since the list is truncated; any shortfall is filled with fresh
        random genomes. `id_offset` starts player_id numbering from a
        non-zero base -- needed when multiple independent Populations
        coexist (see IslandModel below), since each one otherwise starts
        counting from 0 and their player_ids would collide."""
        self.config = config
        self.rng = rng
        self.generation = 0
        self._next_id = id_offset
        self.players: list[Player] = []

        if seed_genomes:
            for genome in seed_genomes[: config.population_size]:
                self.players.append(Player(player_id=self._next_id, genome=genome.copy(), generation=self.generation))
                self._next_id += 1
        while len(self.players) < config.population_size:
            self.players.append(self._new_random_player())

    def _new_random_player(self) -> Player:
        genome = Genome.random(self.rng, scale=self.config.init_scale)
        player = Player(player_id=self._next_id, genome=genome, generation=self.generation)
        self._next_id += 1
        return player

    def evolve(self, fitness: dict[int, float]) -> list[Player]:
        """Given a fitness score per current player_id, produce the next
        generation's player list (does not mutate self.players; caller
        should assign the result)."""
        cfg = self.config
        ranked = sorted(self.players, key=lambda p: fitness.get(p.player_id, 0.0), reverse=True)

        next_gen: list[Player] = []
        for elite in ranked[: cfg.elite_count]:
            child_genome = elite.genome.copy()
            next_gen.append(Player(self._next_id, child_genome, self.generation + 1, elite.label))
            self._next_id += 1

        while len(next_gen) < cfg.population_size:
            parent_a = tournament_select(self.players, fitness, cfg.tournament_size, self.rng)
            if self.rng.random() < cfg.crossover_rate:
                parent_b = tournament_select(self.players, fitness, cfg.tournament_size, self.rng)
                child_genome = crossover(parent_a.genome, parent_b.genome, self.rng)
            else:
                child_genome = parent_a.genome.copy()
            child_genome = mutate(child_genome, cfg.mutation_rate, cfg.mutation_scale, self.rng)
            next_gen.append(Player(self._next_id, child_genome, self.generation + 1))
            self._next_id += 1

        self.generation += 1
        self.players = next_gen
        return next_gen


ISLAND_ID_SPACING = 1_000_000  # headroom per island so player_ids never collide


@dataclass
class IslandConfig:
    num_islands: int = 3  # 1 disables the island model (behaves like a single Population)
    # Generations between migration events. 0 disables migration (islands
    # then evolve fully independently for the whole run).
    migration_interval: int = 10
    # How many of an island's best individuals migrate per event.
    migration_size: int = 3


class IslandModel:
    """Manages `num_islands` independent Populations that both breed AND
    are evaluated in isolation from each other -- each island's players
    only ever share a table with other members of the same island (see
    main.py, which calls run_generation once per island rather than once
    over the combined population). That isolation is the point: if one
    island's fitness landscape gets taken over by a pathological strategy
    (e.g. the "never fold" collapse this project spent a while diagnosing),
    the other islands aren't directly exposed to it as opponents, so they
    don't automatically inherit the same collapse. A ring migration every
    `migration_interval` generations is the only channel connecting
    islands: each island's best `migration_size` genomes this generation
    are copied into the next island in the ring, replacing that island's
    worst-off *slots* -- but not literally its worst-fitness individuals,
    since the newly-bred next generation hasn't been evaluated yet at the
    point migration happens. Random non-elite slots are used instead
    (elitism already protects each island's own best from being
    overwritten), which sidesteps comparing fitness numbers across islands
    that were never measured against the same opponents anyway."""

    def __init__(
        self,
        ga_config: GAConfig,
        island_config: IslandConfig,
        rng: np.random.Generator,
        seed_genomes: list[Genome] | None = None,
    ):
        self.island_config = island_config
        self.rng = rng
        self.generation = 0
        self.islands: list[Population] = []
        for i in range(island_config.num_islands):
            # Round-robin the reloaded population across islands (rather
            # than slicing it into contiguous chunks) so every island gets
            # an even spread of quality, not "island 0 gets all the best."
            seeds = seed_genomes[i :: island_config.num_islands] if seed_genomes else None
            self.islands.append(Population(ga_config, rng, seed_genomes=seeds, id_offset=i * ISLAND_ID_SPACING))

    @property
    def all_players(self) -> list[Player]:
        return [p for island in self.islands for p in island.players]

    def evolve_all(self, fitness_by_island: list[dict[int, float]]) -> None:
        """Given one fitness dict per island (same order as self.islands),
        evolves every island independently, then performs ring migration if
        this generation lands on the migration interval."""
        self.generation += 1
        do_migration = (
            self.island_config.migration_interval > 0
            and self.generation % self.island_config.migration_interval == 0
            and len(self.islands) > 1
        )

        migrants_out = None
        if do_migration:
            k = self.island_config.migration_size
            migrants_out = []
            for island, fitness in zip(self.islands, fitness_by_island):
                ranked = sorted(island.players, key=lambda p: fitness.get(p.player_id, 0.0), reverse=True)
                migrants_out.append([p.genome.copy() for p in ranked[:k]])

        for island, fitness in zip(self.islands, fitness_by_island):
            island.evolve(fitness)

        if do_migration:
            n = len(self.islands)
            for i, island in enumerate(self.islands):
                incoming = migrants_out[(i - 1) % n]  # island i receives from its ring predecessor
                self._splice_in_migrants(island, incoming)

    def _splice_in_migrants(self, island: Population, migrant_genomes: list[Genome]) -> None:
        elite_count = island.config.elite_count
        candidate_idx = list(range(elite_count, len(island.players)))
        chosen = self.rng.choice(
            candidate_idx, size=min(len(migrant_genomes), len(candidate_idx)), replace=False
        )
        for idx, genome in zip(chosen, migrant_genomes):
            island.players[idx] = Player(island._next_id, genome.copy(), island.generation, label="migrant")
            island._next_id += 1
