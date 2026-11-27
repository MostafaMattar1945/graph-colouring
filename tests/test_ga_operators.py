
import random

from src.graph_model import Graph
from src.ga_operators import (
    create_individual,
    create_population,
    conflicts_of,
    fitness,
    tournament_selection,
    roulette_wheel_selection,
    one_point_crossover,
    mutate
)


def create_test_graph():
    return Graph.from_edges(
        ["A", "B", "C", "D"],
        [
            ("A", "B"),
            ("B", "C"),
            ("C", "D")
        ]
    )


def test_create_individual():
    individual = create_individual(
        n_genes=5,
        k=3,
        rng=random.Random(42)
    )

    assert len(individual) == 5
    assert all(0 <= colour < 3 for colour in individual)


def test_create_population():
    population = create_population(
        pop_size=10,
        n_genes=5,
        k=3,
        rng=random.Random(42)
    )

    assert len(population) == 10
    assert all(len(individual) == 5 for individual in population)


def test_conflicts_of():
    g = create_test_graph()

    # A and B have the same colour.
    # C and D have the same colour.
    individual = [0, 0, 1, 1]

    assert conflicts_of(individual, g) == 2


def test_fitness():
    g = create_test_graph()

    individual = [0, 0, 1, 1]

    assert fitness(individual, g) == 1 / 3


def test_tournament_selection():
    population = [
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 0]
    ]

    fitness_fn = lambda individual: 1.0 if individual == [0, 1] else 0.1

    parents = tournament_selection(
        population,
        fitness_fn,
        num_parents=2,
        tournament_size=4,
        rng=random.Random(42)
    )

    assert len(parents) == 2
    assert all(parent in population for parent in parents)


def test_roulette_wheel_selection():
    population = [
        [0, 0],
        [0, 1],
        [1, 1]
    ]

    fitness_fn = lambda individual: 1.0 if individual == [0, 1] else 0.1

    parents = roulette_wheel_selection(
        population,
        fitness_fn,
        num_parents=5,
        rng=random.Random(42)
    )

    assert len(parents) == 5
    assert all(parent in population for parent in parents)


def test_one_point_crossover():
    parent1 = [0, 0, 0, 0]
    parent2 = [1, 1, 1, 1]

    child1, child2 = one_point_crossover(
        parent1,
        parent2,
        rng=random.Random(42)
    )

    assert len(child1) == len(parent1)
    assert len(child2) == len(parent2)
    assert child1 != parent1
    assert child2 != parent2


def test_mutate_changes_gene():
    individual = [0, 0, 0, 0]

    mutated = mutate(
        individual,
        k=2,
        mutation_rate=1.0,
        rng=random.Random(42)
    )

    assert mutated != individual
    assert all(colour in [0, 1] for colour in mutated)


def test_mutate_does_not_change_original():
    individual = [0, 0, 0, 0]

    mutated = mutate(
        individual,
        k=2,
        mutation_rate=1.0,
        rng=random.Random(42)
    )

    assert individual == [0, 0, 0, 0]
    assert mutated is not individual
