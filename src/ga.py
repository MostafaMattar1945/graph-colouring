
import random
import time

from src.common import GAParams, Result
from src.graph_model import greedy_upper_bound, clique_lower_bound

from src.ga_operators import (
    create_population,
    conflicts_of,
    fitness,
    tournament_selection,
    roulette_wheel_selection,
    one_point_crossover,
    mutate
)


def solve_ga_fixed_k(g, k, params, rng=None):
    """
    Run Genetic Algorithm using a fixed number of colours.
    """

    if rng is None:
        rng = random.Random(params.seed)

    n = len(g.vertices)
    fitness_fn = lambda individual: fitness(individual, g)

    population = create_population(params.pop_size, n, k, rng)

    history = []
    best_conflicts = float("inf")
    best_individual = None
    stale_generations = 0
    generations_used = 0

    elite_count = max(0, min(params.elite_count, params.pop_size))

    for generation in range(params.generations):
        generations_used += 1

        population.sort(
            key=lambda individual: (
                fitness_fn(individual),
                -conflicts_of(individual, g)
            ),
            reverse=True
        )

        current_best = population[0]
        current_conflicts = conflicts_of(current_best, g)

        if current_conflicts < best_conflicts:
            best_conflicts = current_conflicts
            best_individual = current_best.copy()
            stale_generations = 0
        else:
            stale_generations += 1

        history.append(best_conflicts)

        if best_conflicts == 0:
            break

        if params.patience > 0 and stale_generations >= params.patience:
            break

        new_population = [
            individual.copy()
            for individual in population[:elite_count]
        ]

        while len(new_population) < params.pop_size:

            if params.selection == "tournament":
                parents = tournament_selection(
                    population,
                    fitness_fn,
                    2,
                    tournament_size=min(
                        params.tournament_size,
                        len(population)
                    ),
                    rng=rng
                )

            elif params.selection == "roulette":
                parents = roulette_wheel_selection(
                    population,
                    fitness_fn,
                    2,
                    rng
                )

            else:
                raise ValueError(
                    "Selection must be 'tournament' or 'roulette'."
                )

            parent1, parent2 = parents

            if rng.random() < params.crossover_rate:
                child1, child2 = one_point_crossover(
                    parent1, parent2, rng
                )
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()

            child1 = mutate(
                child1, k, params.mutation_rate, rng
            )
            child2 = mutate(
                child2, k, params.mutation_rate, rng
            )

            new_population.append(child1)

            if len(new_population) < params.pop_size:
                new_population.append(child2)

        population = new_population

    population.sort(
        key=lambda individual: (
            fitness_fn(individual),
            -conflicts_of(individual, g)
        ),
        reverse=True
    )

    final_best = population[0]
    final_conflicts = conflicts_of(final_best, g)

    if final_conflicts < best_conflicts:
        best_conflicts = final_conflicts
        best_individual = final_best.copy()

        if history:
            history[-1] = best_conflicts

    return best_individual, history, generations_used


def solve_ga(g, params: GAParams) -> Result:
    start_time = time.perf_counter()

    if len(g.vertices) == 0:
        return Result(
            algorithm="GA",
            success=True,
            colouring={},
            k_used=0,
            conflicts=0,
            time_ms=(time.perf_counter() - start_time) * 1000,
            iterations=0,
            history=[],
            log=["Empty graph: no colouring required."]
        )

    if params.pop_size < 1:
        raise ValueError("pop_size must be at least 1.")

    if params.generations < 1:
        raise ValueError("generations must be at least 1.")

    lower = clique_lower_bound(g)
    upper = greedy_upper_bound(g)

    rng = random.Random(params.seed)

    total_generations = 0
    log = []

    best_overall = None
    best_overall_conflicts = float("inf")
    best_overall_k = None
    best_overall_history = []

    for k in range(lower, upper + 1):
        log.append(f"Trying GA with k={k}.")

        best_individual, history, generations_used = solve_ga_fixed_k(
            g, k, params, rng
        )

        total_generations += generations_used
        current_conflicts = conflicts_of(best_individual, g)

        if current_conflicts < best_overall_conflicts:
            best_overall = best_individual.copy()
            best_overall_conflicts = current_conflicts
            best_overall_k = k
            best_overall_history = history.copy()

        if current_conflicts == 0:
            colouring = dict(zip(g.vertices, best_individual))

            if g.is_valid(colouring):
                log.append(f"Valid colouring found using k={k}.")

                return Result(
                    algorithm="GA",
                    success=True,
                    colouring=colouring,
                    k_used=k,
                    conflicts=0,
                    time_ms=(time.perf_counter() - start_time) * 1000,
                    iterations=total_generations,
                    history=history,
                    log=log
                )

    log.append(
        "No valid colouring found. GA is heuristic, "
        "so this does not prove that no solution exists."
    )

    return Result(
        algorithm="GA",
        success=False,
        colouring=None,
        k_used=best_overall_k,
        conflicts=best_overall_conflicts,
        time_ms=(time.perf_counter() - start_time) * 1000,
        iterations=total_generations,
        history=best_overall_history,
        log=log
    )
