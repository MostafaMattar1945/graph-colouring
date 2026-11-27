
import src.ga as ga_module

from src.common import GAParams
from src.graph_model import Graph
from src.ga import solve_ga


def create_k5_graph():
    return Graph.from_edges(
        list("ABCDE"),
        [
            ("A", "B"), ("A", "C"), ("A", "D"), ("A", "E"),
            ("B", "C"), ("B", "D"), ("B", "E"),
            ("C", "D"), ("C", "E"),
            ("D", "E")
        ]
    )


def create_bipartite_graph():
    return Graph.from_edges(
        list("ABCDEF"),
        [
            ("A", "D"), ("A", "E"), ("A", "F"),
            ("B", "D"), ("B", "E"), ("B", "F"),
            ("C", "D"), ("C", "E"), ("C", "F")
        ]
    )


def test_ga_solves_complete_graph_k5():
    g = create_k5_graph()

    params = GAParams(
        pop_size=50,
        generations=100,
        mutation_rate=0.05,
        seed=42
    )

    result = solve_ga(g, params)

    assert result.success is True
    assert result.algorithm == "GA"
    assert result.conflicts == 0
    assert result.colouring is not None
    assert g.is_valid(result.colouring)
    assert result.k_used == 5


def test_ga_solves_bipartite_graph():
    g = create_bipartite_graph()

    params = GAParams(
        pop_size=50,
        generations=100,
        mutation_rate=0.05,
        seed=42
    )

    result = solve_ga(g, params)

    assert result.success is True
    assert result.conflicts == 0
    assert result.colouring is not None
    assert g.is_valid(result.colouring)
    assert result.k_used == 2


def test_ga_reproducibility_with_same_seed():
    g = create_bipartite_graph()

    params = GAParams(
        pop_size=50,
        generations=100,
        mutation_rate=0.05,
        seed=42
    )

    result1 = solve_ga(g, params)
    result2 = solve_ga(g, params)

    assert result1.success == result2.success
    assert result1.colouring == result2.colouring
    assert result1.conflicts == result2.conflicts
    assert result1.k_used == result2.k_used
    assert result1.history == result2.history


def test_ga_with_roulette_selection():
    g = create_bipartite_graph()

    params = GAParams(
        pop_size=50,
        generations=100,
        mutation_rate=0.05,
        selection="roulette",
        seed=42
    )

    result = solve_ga(g, params)

    assert result.success is True
    assert result.conflicts == 0
    assert result.colouring is not None
    assert g.is_valid(result.colouring)


def test_ga_failure_returns_none_colouring(monkeypatch):
    g = create_k5_graph()

    params = GAParams(
        pop_size=20,
        generations=30,
        mutation_rate=0.05,
        seed=42
    )

    # Force the fixed-k solver to return an invalid colouring.
    def fake_solve_ga_fixed_k(g, k, params, rng=None):
        individual = [0] * len(g.vertices)
        conflicts = g.count_conflicts(
            dict(zip(g.vertices, individual))
        )

        return individual, [conflicts], 1

    monkeypatch.setattr(
        ga_module,
        "solve_ga_fixed_k",
        fake_solve_ga_fixed_k
    )

    result = solve_ga(g, params)

    assert result.success is False
    assert result.colouring is None
    assert result.conflicts > 0
    assert result.k_used == 5


def test_ga_empty_graph():
    g = Graph()

    params = GAParams(seed=42)

    result = solve_ga(g, params)

    assert result.success is True
    assert result.algorithm == "GA"
    assert result.colouring == {}
    assert result.k_used == 0
    assert result.conflicts == 0
    assert result.iterations == 0
