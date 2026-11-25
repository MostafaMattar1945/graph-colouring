import pytest

from src.common import GAParams, Result
from src.graph_model import (
    EXPECTED_CHI,
    Graph,
    GraphError,
    clique_lower_bound,
    greedy_upper_bound,
    list_presets,
    load_preset,
    parse_csv,
    parse_dimacs,
    parse_text,
    random_graph,
)


def test_undirected_no_duplicates_no_self_loop():
    g = Graph.from_edges(["A", "B", "C"], [("A", "B"), ("B", "A"), ("B", "C")])
    assert g.degree("A") == 1
    assert set(g.edges) == {("A", "B"), ("B", "C")}
    with pytest.raises(GraphError, match="Self-loops"):
        g.add_edge("C", "C")


def test_isolated_vertex_and_undefined():
    g = Graph.from_edges(["A", "B", "Z"], [("A", "B")])
    assert "Z" in g.vertices
    assert g.degree("Z") == 0
    with pytest.raises(GraphError, match="not defined"):
        g.degree("Q")


def test_neighbors_sorted():
    g = Graph.from_edges(["C", "A", "B"], [("C", "B"), ("C", "A")])
    assert g.neighbors("C") == ["A", "B"]


def test_copy_is_independent():
    g = Graph.from_edges(["A", "B"], [("A", "B")])
    h = g.copy()
    h.add_vertex("C")
    h.add_edge("A", "C")
    assert "C" not in g.vertices
    assert g.degree("A") == 1


def test_count_conflicts_and_is_valid():
    g = load_preset("exam_schedule")
    valid = {"A": 1, "B": 2, "C": 0, "D": 2, "E": 1, "F": 0, "G": 2}
    assert g.count_conflicts(valid) == 0
    assert g.is_valid(valid)
    bad = dict(valid)
    bad["B"] = 0
    assert g.count_conflicts(bad) > 0
    assert not g.is_valid(bad)
    incomplete = {"A": 0}
    assert not g.is_valid(incomplete)


def test_parsers():
    text = parse_text("# comment\nA B\nB C\nD\n")
    assert set(text.vertices) == {"A", "B", "C", "D"}
    csv_g = parse_csv("u,v\nA,B\nB,C\n")
    assert ("A", "B") in csv_g.edges
    dimacs = parse_dimacs("c demo\np edge 3 2\ne 1 2\ne 2 3\n")
    assert dimacs.vertices == ["1", "2", "3"]
    with pytest.raises(GraphError, match="empty"):
        parse_text("\n# only comments\n")
    with pytest.raises(GraphError, match="not defined"):
        parse_dimacs("p edge 2 1\ne 1 9\n")


def test_random_graph_seeded():
    a = random_graph(12, 0.4, seed=42)
    b = random_graph(12, 0.4, seed=42)
    c = random_graph(12, 0.4, seed=7)
    assert a.edges == b.edges
    assert a.edges != c.edges


def test_bounds_bracket_known_chi():
    from src.graph_model import _greedy_colouring

    for name, chi in EXPECTED_CHI.items():
        g = load_preset(name)
        lo = clique_lower_bound(g)
        hi = greedy_upper_bound(g)
        assert lo <= chi <= hi, (name, lo, chi, hi)
        assert g.is_valid(_greedy_colouring(g))


def test_preset_sizes():
    exam = load_preset("exam_schedule")
    assert len(exam.vertices) == 7
    assert len(exam.edges) == 11
    assert clique_lower_bound(exam) >= 3
    k5 = load_preset("complete_k5")
    assert clique_lower_bound(k5) == 5
    assert greedy_upper_bound(k5) == 5
    k33 = load_preset("k33")
    assert clique_lower_bound(k33) == 2
    grotzsch = load_preset("grotzsch")
    assert len(grotzsch.vertices) == 11
    assert clique_lower_bound(grotzsch) == 2  # triangle-free


def test_list_presets_and_unknown():
    assert list_presets() == list(EXPECTED_CHI)
    with pytest.raises(GraphError, match="Unknown preset"):
        load_preset("not-a-graph")


def test_contract_dataclasses():
    p = GAParams()
    assert p.pop_size == 50 and p.mutation_rate == 0.05 and p.ls_rate == 1.0
    r = Result(
        algorithm="Backtracking",
        success=False,
        colouring=None,
        k_used=None,
        conflicts=3,
        time_ms=1.0,
        iterations=0,
    )
    assert r.backtracks == 0 and r.history == [] and r.timed_out is False


def test_myciel_alias_is_grotzsch_only_for_myciel3():
    g = load_preset("myciel3")  # DIMACS myciel3 = Grotzsch: 11 vertices, 20 edges
    assert len(g.vertices) == 11 and len(g.edges) == 20
    with pytest.raises(GraphError, match="Unknown preset"):
        load_preset("myciel4")  # DIMACS myciel4 has 23 vertices, so it is NOT a preset