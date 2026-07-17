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
    def __init__(self, config: GAConfig, rng: np.random.Generator, seed_genomes: list[Genome] | None = None):
        """`seed_genomes`, if given, seeds generation 0 (e.g. reloaded from a
        previous run's final population) instead of starting fully random.
        Best-first ordering matters if len(seed_genomes) > population_size,
        since the list is truncated; any shortfall is filled with fresh
        random genomes."""
        self.config = config
        self.rng = rng
        self.generation = 0
        self._next_id = 0
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
