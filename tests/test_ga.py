import numpy as np
import pytest

from ga import (
    ISLAND_ID_SPACING, GAConfig, IslandConfig, IslandModel, Population,
    crossover, mutate, tournament_select,
)
from genome import Genome
from player import Player


def make_players(n, id_offset=0):
    rng = np.random.default_rng(0)
    return [Player(player_id=id_offset + i, genome=Genome.random(rng)) for i in range(n)]


class TestTournamentSelect:
    def test_returns_global_best_when_all_are_contestants(self):
        players = make_players(6)
        fitness = {p.player_id: float(i) for i, p in enumerate(players)}
        rng = np.random.default_rng(0)
        winner = tournament_select(players, fitness, k=len(players), rng=rng)
        assert winner.player_id == players[-1].player_id  # highest fitness

    def test_returned_player_is_one_of_the_contestants(self):
        players = make_players(10)
        fitness = {p.player_id: float(np.random.default_rng(i).random()) for i, p in enumerate(players)}
        rng = np.random.default_rng(1)
        for _ in range(20):
            winner = tournament_select(players, fitness, k=3, rng=rng)
            assert winner in players


class TestCrossoverAndMutateDelegation:
    def test_crossover_produces_a_new_genome(self):
        rng = np.random.default_rng(0)
        a, b = Genome.random(rng), Genome.random(rng)
        child = crossover(a, b, rng)
        assert isinstance(child, Genome)
        assert child is not a and child is not b

    def test_mutate_produces_a_new_genome(self):
        rng = np.random.default_rng(0)
        g = Genome.random(rng)
        mutated = mutate(g, rate=1.0, scale=1.0, rng=rng)
        assert isinstance(mutated, Genome)
        assert mutated is not g


class TestPopulationInit:
    def test_random_population_has_configured_size(self):
        cfg = GAConfig(population_size=12)
        pop = Population(cfg, np.random.default_rng(0))
        assert len(pop.players) == 12

    def test_player_ids_are_sequential_from_offset(self):
        cfg = GAConfig(population_size=6)
        pop = Population(cfg, np.random.default_rng(0), id_offset=100)
        ids = sorted(p.player_id for p in pop.players)
        assert ids == list(range(100, 106))

    def test_seed_genomes_shorter_than_population_are_padded_randomly(self):
        cfg = GAConfig(population_size=6)
        seeds = [Genome.random(np.random.default_rng(0)) for _ in range(2)]
        pop = Population(cfg, np.random.default_rng(1), seed_genomes=seeds)
        assert len(pop.players) == 6
        assert np.array_equal(pop.players[0].genome.weights_v, seeds[0].weights_v)
        assert np.array_equal(pop.players[1].genome.weights_v, seeds[1].weights_v)

    def test_seed_genomes_longer_than_population_are_truncated(self):
        cfg = GAConfig(population_size=3)
        seeds = [Genome.random(np.random.default_rng(i)) for i in range(6)]
        pop = Population(cfg, np.random.default_rng(1), seed_genomes=seeds)
        assert len(pop.players) == 3

    def test_seed_genomes_are_copied_not_shared(self):
        cfg = GAConfig(population_size=2)
        seeds = [Genome.random(np.random.default_rng(0)) for _ in range(2)]
        pop = Population(cfg, np.random.default_rng(1), seed_genomes=seeds)
        pop.players[0].genome.weights_v[0] = 999.0
        assert seeds[0].weights_v[0] != 999.0


class TestPopulationEvolve:
    def test_next_generation_has_same_population_size(self):
        cfg = GAConfig(population_size=12, elite_count=2)
        pop = Population(cfg, np.random.default_rng(0))
        fitness = {p.player_id: float(i) for i, p in enumerate(pop.players)}
        next_gen = pop.evolve(fitness)
        assert len(next_gen) == 12

    def test_generation_counter_increments(self):
        cfg = GAConfig(population_size=6)
        pop = Population(cfg, np.random.default_rng(0))
        fitness = {p.player_id: 0.0 for p in pop.players}
        pop.evolve(fitness)
        assert pop.generation == 1

    def test_elites_are_copies_of_the_fittest_players(self):
        cfg = GAConfig(population_size=6, elite_count=2)
        pop = Population(cfg, np.random.default_rng(0))
        fitness = {p.player_id: float(i) for i, p in enumerate(pop.players)}
        best_two = sorted(pop.players, key=lambda p: fitness[p.player_id], reverse=True)[:2]
        next_gen = pop.evolve(fitness)
        elites_in_next_gen = next_gen[:2]
        for elite, original in zip(elites_in_next_gen, best_two):
            assert np.array_equal(elite.genome.weights_v, original.genome.weights_v)
            assert elite.genome is not original.genome  # copied, not shared

    def test_new_player_ids_are_unique_and_continue_counting(self):
        cfg = GAConfig(population_size=6)
        pop = Population(cfg, np.random.default_rng(0))
        fitness = {p.player_id: 0.0 for p in pop.players}
        first_gen_ids = {p.player_id for p in pop.players}
        next_gen = pop.evolve(fitness)
        next_gen_ids = {p.player_id for p in next_gen}
        assert len(next_gen_ids) == 6
        assert first_gen_ids.isdisjoint(next_gen_ids)

    def test_zero_elites_still_produces_a_full_generation(self):
        cfg = GAConfig(population_size=6, elite_count=0)
        pop = Population(cfg, np.random.default_rng(0))
        fitness = {p.player_id: float(i) for i, p in enumerate(pop.players)}
        next_gen = pop.evolve(fitness)
        assert len(next_gen) == 6

    def test_players_attribute_reflects_new_generation(self):
        cfg = GAConfig(population_size=6)
        pop = Population(cfg, np.random.default_rng(0))
        fitness = {p.player_id: 0.0 for p in pop.players}
        next_gen = pop.evolve(fitness)
        assert pop.players == next_gen


class TestIslandModel:
    def test_islands_each_get_configured_population_size(self):
        ga_cfg = GAConfig(population_size=6)
        island_cfg = IslandConfig(num_islands=3, migration_interval=0)
        model = IslandModel(ga_cfg, island_cfg, np.random.default_rng(0))
        assert len(model.islands) == 3
        for island in model.islands:
            assert len(island.players) == 6

    def test_player_ids_dont_collide_across_islands(self):
        ga_cfg = GAConfig(population_size=6)
        island_cfg = IslandConfig(num_islands=3, migration_interval=0)
        model = IslandModel(ga_cfg, island_cfg, np.random.default_rng(0))
        ids = [p.player_id for p in model.all_players]
        assert len(ids) == len(set(ids))

    def test_island_id_offsets_use_configured_spacing(self):
        ga_cfg = GAConfig(population_size=6)
        island_cfg = IslandConfig(num_islands=3, migration_interval=0)
        model = IslandModel(ga_cfg, island_cfg, np.random.default_rng(0))
        for i, island in enumerate(model.islands):
            ids = [p.player_id for p in island.players]
            assert min(ids) >= i * ISLAND_ID_SPACING
            assert max(ids) < (i + 1) * ISLAND_ID_SPACING

    def test_all_players_combines_every_island(self):
        ga_cfg = GAConfig(population_size=6)
        island_cfg = IslandConfig(num_islands=2, migration_interval=0)
        model = IslandModel(ga_cfg, island_cfg, np.random.default_rng(0))
        assert len(model.all_players) == 12

    def test_seed_genomes_are_spread_round_robin_across_islands(self):
        ga_cfg = GAConfig(population_size=6)
        island_cfg = IslandConfig(num_islands=2, migration_interval=0)
        seeds = [Genome.random(np.random.default_rng(i)) for i in range(12)]
        model = IslandModel(ga_cfg, island_cfg, np.random.default_rng(0), seed_genomes=seeds)
        island0_weights = {tuple(p.genome.weights_v) for p in model.islands[0].players}
        island1_weights = {tuple(p.genome.weights_v) for p in model.islands[1].players}
        expected0 = {tuple(seeds[i].weights_v) for i in range(0, 12, 2)}
        expected1 = {tuple(seeds[i].weights_v) for i in range(1, 12, 2)}
        assert island0_weights == expected0
        assert island1_weights == expected1

    def test_evolve_all_increments_generation(self):
        ga_cfg = GAConfig(population_size=6)
        island_cfg = IslandConfig(num_islands=2, migration_interval=0)
        model = IslandModel(ga_cfg, island_cfg, np.random.default_rng(0))
        fitness_by_island = [{p.player_id: 0.0 for p in island.players} for island in model.islands]
        model.evolve_all(fitness_by_island)
        assert model.generation == 1

    def test_no_migration_when_disabled(self):
        ga_cfg = GAConfig(population_size=6)
        island_cfg = IslandConfig(num_islands=2, migration_interval=0)
        model = IslandModel(ga_cfg, island_cfg, np.random.default_rng(0))
        fitness_by_island = [{p.player_id: float(i) for i, p in enumerate(isl.players)} for isl in model.islands]
        model.evolve_all(fitness_by_island)
        for island in model.islands:
            assert all(p.label != "migrant" for p in island.players)

    def test_no_migration_with_a_single_island(self):
        ga_cfg = GAConfig(population_size=6)
        island_cfg = IslandConfig(num_islands=1, migration_interval=1, migration_size=2)
        model = IslandModel(ga_cfg, island_cfg, np.random.default_rng(0))
        fitness_by_island = [{p.player_id: float(i) for i, p in enumerate(model.islands[0].players)}]
        model.evolve_all(fitness_by_island)
        assert all(p.label != "migrant" for p in model.islands[0].players)

    def test_migration_happens_on_the_configured_interval(self):
        ga_cfg = GAConfig(population_size=12, elite_count=2)
        island_cfg = IslandConfig(num_islands=2, migration_interval=1, migration_size=2)
        model = IslandModel(ga_cfg, island_cfg, np.random.default_rng(0))
        fitness_by_island = [{p.player_id: float(i) for i, p in enumerate(isl.players)} for isl in model.islands]
        model.evolve_all(fitness_by_island)
        for island in model.islands:
            migrants = [p for p in island.players if p.label == "migrant"]
            assert len(migrants) == island_cfg.migration_size

    def test_migrants_never_overwrite_elite_slots(self):
        ga_cfg = GAConfig(population_size=12, elite_count=4)
        island_cfg = IslandConfig(num_islands=2, migration_interval=1, migration_size=4)
        model = IslandModel(ga_cfg, island_cfg, np.random.default_rng(0))
        fitness_by_island = [{p.player_id: float(i) for i, p in enumerate(isl.players)} for isl in model.islands]
        model.evolve_all(fitness_by_island)
        for island in model.islands:
            elite_slots = island.players[: ga_cfg.elite_count]
            assert all(p.label != "migrant" for p in elite_slots)


class TestForceGtoIslands:
    def test_disabled_by_default_leaves_gto_flags_free(self):
        ga_cfg = GAConfig(population_size=6)
        island_cfg = IslandConfig(num_islands=2, migration_interval=0)
        model = IslandModel(ga_cfg, island_cfg, np.random.default_rng(0))
        # With GTO_INIT_PROB well under 1, a freshly random population of 12
        # genomes practically never lands on "every flag active" by chance.
        assert not all(np.all(p.genome.gto_flags > 0.5) for p in model.all_players)

    def test_forced_islands_start_with_every_gto_flag_active(self):
        ga_cfg = GAConfig(population_size=6)
        island_cfg = IslandConfig(num_islands=3, migration_interval=0, force_gto_islands=2)
        model = IslandModel(ga_cfg, island_cfg, np.random.default_rng(0))
        for island in model.islands[:2]:
            for p in island.players:
                assert np.all(p.genome.gto_flags > 0.5)

    def test_non_forced_islands_are_unaffected(self):
        ga_cfg = GAConfig(population_size=6)
        island_cfg = IslandConfig(num_islands=3, migration_interval=0, force_gto_islands=2)
        model = IslandModel(ga_cfg, island_cfg, np.random.default_rng(0))
        assert not all(np.all(p.genome.gto_flags > 0.5) for p in model.islands[2].players)

    def test_forced_flags_survive_evolution(self):
        ga_cfg = GAConfig(population_size=12, elite_count=2, mutation_rate=1.0)
        island_cfg = IslandConfig(num_islands=2, migration_interval=0, force_gto_islands=1)
        model = IslandModel(ga_cfg, island_cfg, np.random.default_rng(0))
        fitness_by_island = [{p.player_id: float(i) for i, p in enumerate(isl.players)} for isl in model.islands]
        model.evolve_all(fitness_by_island)
        for p in model.islands[0].players:
            assert np.all(p.genome.gto_flags > 0.5)

    def test_forced_flags_survive_migration_into_a_forced_island(self):
        ga_cfg = GAConfig(population_size=12, elite_count=2)
        island_cfg = IslandConfig(num_islands=2, migration_interval=1, migration_size=4, force_gto_islands=1)
        model = IslandModel(ga_cfg, island_cfg, np.random.default_rng(0))
        # Island 1 (not forced) should breed migrants with a realistic mix
        # of active/inactive flags; island 0 (forced, ring-receives from
        # island 1) must still end up all-on despite that.
        fitness_by_island = [{p.player_id: float(i) for i, p in enumerate(isl.players)} for isl in model.islands]
        model.evolve_all(fitness_by_island)
        for p in model.islands[0].players:
            assert np.all(p.genome.gto_flags > 0.5)
