from __future__ import annotations

import csv
import io
import random
import re
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from .common import Colouring

Edge = Tuple[str, str]


class GraphError(ValueError):
    """Invalid graph input or operation."""


class Graph:
    """Undirected simple graph: string vertices, no self-loops, no duplicate edges."""

    def __init__(self) -> None:
        self._adj: Dict[str, Set[str]] = {}  # a dict keeps insertion order
        self._edges: List[Edge] = []         # every edge once, in the order added

    @property
    def vertices(self) -> List[str]:
        return list(self._adj)

    @property
    def adj(self) -> Dict[str, Set[str]]:
        return {v: set(nbrs) for v, nbrs in self._adj.items()}

    @property
    def edges(self) -> List[Edge]:
        return list(self._edges)

    @classmethod
    def from_edges(cls, vertices: List[str], edges: List[Edge]) -> "Graph":
        g = cls()
        for v in vertices:
            g.add_vertex(v)
        for u, v in edges:
            g.add_edge(u, v)
        return g

    def add_vertex(self, v: str) -> None:
        self._adj.setdefault(str(v), set())

    def add_edge(self, u: str, v: str) -> None:
        u, v = str(u), str(v)
        if u == v:
            raise GraphError(f"Self-loops make colouring impossible ({u}-{v}). Please remove it.")
        self.add_vertex(u)
        self.add_vertex(v)
        if v not in self._adj[u]:  # ignore duplicates
            self._adj[u].add(v)
            self._adj[v].add(u)
            self._edges.append((u, v))

    def copy(self) -> "Graph":
        return Graph.from_edges(self.vertices, self._edges)

    def _check(self, v: str) -> str:
        v = str(v)
        if v not in self._adj:
            raise GraphError(f"Vertex {v} is not defined.")
        return v

    def degree(self, v: str) -> int:
        return len(self._adj[self._check(v)])

    def neighbors(self, v: str) -> List[str]:
        return sorted(self._adj[self._check(v)])

    def count_conflicts(self, colouring: Colouring) -> int:
        return sum(
            1 for u, v in self._edges
            if u in colouring and v in colouring and colouring[u] == colouring[v]
        )

    def is_valid(self, colouring: Colouring) -> bool:
        if colouring is None or any(v not in colouring for v in self._adj):
            return False
        return self.count_conflicts(colouring) == 0


_COMMENT = re.compile(r"^\s*(#|//)")


def _nonempty_lines(text: str) -> Iterable[str]:
    for raw in text.splitlines():
        line = raw.strip()
        if not line or _COMMENT.match(line):
            continue
        yield line


def _require_nonempty(g: Graph) -> Graph:
    if not g.vertices:
        raise GraphError("Graph is empty: add at least one vertex.")
    return g


def parse_text(text: str) -> Graph:
    """Plain text, one edge per line: 'A B'. A single token adds an isolated vertex."""
    g = Graph()
    for line in _nonempty_lines(text):
        parts = line.replace(",", " ").split()
        if len(parts) == 1 and "-" in parts[0]:  # "A-B" shorthand for an edge
            parts = parts[0].split("-")
        if len(parts) == 1:
            g.add_vertex(parts[0])
        elif len(parts) == 2:
            g.add_edge(parts[0], parts[1])
        else:
            raise GraphError(f"Could not parse line (expected 'u v'): {line!r}")
    return _require_nonempty(g)


def parse_csv(text: str) -> Graph:
    """CSV, one edge per line: 'u,v'."""
    g = Graph()
    reader = csv.reader(io.StringIO(text))
    header_skipped = False
    for row in reader:
        cells = [c.strip() for c in row if c.strip() != ""]
        if not cells:
            continue
        if not header_skipped and len(cells) >= 2:
            joined = ",".join(c.lower() for c in cells[:2])
            if joined in {"u,v", "source,target", "from,to", "v1,v2", "vertex,neighbor"}:
                header_skipped = True
                continue
        header_skipped = True
        if len(cells) == 1:
            g.add_vertex(cells[0])
        else:
            g.add_edge(cells[0], cells[1])
    return _require_nonempty(g)


def parse_dimacs(text: str) -> Graph:
    """DIMACS .col: 'p edge n m' header, then 'e u v' lines (1-based vertex ids)."""
    g = Graph()
    n_declared: Optional[int] = None
    for line in _nonempty_lines(text):
        if line.startswith("c") and (len(line) == 1 or line[1].isspace()):
            continue
        parts = line.split()
        kind = parts[0].lower()
        if kind == "p":
            if len(parts) < 4:
                raise GraphError(f"Invalid DIMACS problem line: {line!r}")
            try:
                n_declared = int(parts[2])
                int(parts[3])
            except ValueError as exc:
                raise GraphError(f"Invalid DIMACS problem line: {line!r}") from exc
            if n_declared < 1:
                raise GraphError("Graph is empty: add at least one vertex.")
            for i in range(1, n_declared + 1):
                g.add_vertex(str(i))
            continue
        if kind == "e":
            if len(parts) < 3:
                raise GraphError(f"Invalid DIMACS edge line: {line!r}")
            u, v = parts[1], parts[2]
            if n_declared is not None:
                try:
                    iu, iv = int(u), int(v)
                except ValueError as exc:
                    raise GraphError(f"Vertex {u} is not defined.") from exc
                if iu < 1 or iu > n_declared:
                    raise GraphError(f"Vertex {u} is not defined.")
                if iv < 1 or iv > n_declared:
                    raise GraphError(f"Vertex {v} is not defined.")
            g.add_edge(u, v)
            continue
        raise GraphError(f"Unrecognised DIMACS line: {line!r}")
    return _require_nonempty(g)


def random_graph(n: int, p: float, seed: Optional[int] = None) -> Graph:
    """Seeded Erdos-Renyi G(n, p). Vertices are '0' .. 'n-1'."""
    if n < 1:
        raise GraphError("Graph is empty: add at least one vertex.")
    if not 0.0 <= p <= 1.0:
        raise GraphError("Edge probability p must be in [0, 1].")
    rng = random.Random(seed)
    g = Graph()
    verts = [str(i) for i in range(n)]
    for v in verts:
        g.add_vertex(v)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                g.add_edge(verts[i], verts[j])
    return g


def _greedy_colouring(g: Graph) -> Dict[str, int]:
    order = sorted(g.vertices, key=lambda v: (-g.degree(v), v))
    colouring: Dict[str, int] = {}
    for v in order:
        used = {colouring[n] for n in g.neighbors(v) if n in colouring}
        c = 0
        while c in used:
            c += 1
        colouring[v] = c
    return colouring


def greedy_upper_bound(g: Graph) -> int:
    """Number of colours used by a greedy colouring (a valid upper bound on chi)."""
    if not g.vertices:
        return 0
    colouring = _greedy_colouring(g)
    return max(colouring.values()) + 1


def clique_lower_bound(g: Graph) -> int:
    """Size of a clique found in g (a valid lower bound on chi)."""
    if not g.vertices:
        return 0
    best = 1
    starts = sorted(g.vertices, key=lambda v: (-g.degree(v), v))
    for start in starts:
        clique = [start]
        candidates = set(g.neighbors(start))
        while candidates:
            v = max(candidates, key=lambda x: (g.degree(x), x))
            clique.append(v)
            candidates.intersection_update(g.neighbors(v))
            candidates.discard(v)
        best = max(best, len(clique))
        if best >= len(g.vertices):
            break
    return best


EXPECTED_CHI: Dict[str, int] = {
    "exam_schedule": 3,
    "petersen": 3,
    "cycle_c5": 3,
    "complete_k5": 5,
    "bipartite_k33": 2,
    "grotzsch": 4,
}


def _exam_schedule() -> Graph:
    return Graph.from_edges(
        ["A", "B", "C", "D", "E", "F", "G"],
        [
            ("A", "B"),
            ("A", "C"),
            ("A", "D"),
            ("B", "C"),
            ("B", "E"),
            ("C", "D"),
            ("C", "E"),
            ("D", "F"),
            ("E", "F"),
            ("E", "G"),
            ("F", "G"),
        ],
    )


def _petersen() -> Graph:
    verts = [str(i) for i in range(10)]
    edges: List[Edge] = [
        ("0", "1"),
        ("1", "2"),
        ("2", "3"),
        ("3", "4"),
        ("4", "0"),
        ("0", "5"),
        ("1", "6"),
        ("2", "7"),
        ("3", "8"),
        ("4", "9"),
        ("5", "7"),
        ("7", "9"),
        ("9", "6"),
        ("6", "8"),
        ("8", "5"),
    ]
    return Graph.from_edges(verts, edges)


def _cycle_c5() -> Graph:
    return Graph.from_edges(
        ["A", "B", "C", "D", "E"],
        [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "A")],
    )


def _complete_k5() -> Graph:
    verts = ["A", "B", "C", "D", "E"]
    edges = [(verts[i], verts[j]) for i in range(5) for j in range(i + 1, 5)]
    return Graph.from_edges(verts, edges)


def _bipartite_k33() -> Graph:
    left, right = ["A", "B", "C"], ["D", "E", "F"]
    edges = [(u, v) for u in left for v in right]
    return Graph.from_edges(left + right, edges)


def _grotzsch() -> Graph:
    """Mycielski construction on C5: 11 vertices, triangle-free, χ = 4."""
    c5 = ["0", "1", "2", "3", "4"]
    copies = ["5", "6", "7", "8", "9"]
    w = "10"
    verts = c5 + copies + [w]
    edges: List[Edge] = [
        ("0", "1"),
        ("1", "2"),
        ("2", "3"),
        ("3", "4"),
        ("4", "0"),
        ("5", "1"),
        ("5", "4"),
        ("6", "0"),
        ("6", "2"),
        ("7", "1"),
        ("7", "3"),
        ("8", "2"),
        ("8", "4"),
        ("9", "3"),
        ("9", "0"),
        ("10", "5"),
        ("10", "6"),
        ("10", "7"),
        ("10", "8"),
        ("10", "9"),
    ]
    return Graph.from_edges(verts, edges)


_PRESETS = {
    "exam_schedule": _exam_schedule,
    "petersen": _petersen,
    "cycle_c5": _cycle_c5,
    "complete_k5": _complete_k5,
    "bipartite_k33": _bipartite_k33,
    "grotzsch": _grotzsch,
}

_PRESET_ALIASES = {
    "exam": "exam_schedule",
    "exams": "exam_schedule",
    "petersen_graph": "petersen",
    "c5": "cycle_c5",
    "cycle5": "cycle_c5",
    "k5": "complete_k5",
    "complete5": "complete_k5",
    "k33": "bipartite_k33",
    "k3,3": "bipartite_k33",
    "bipartite": "bipartite_k33",
    "grötzsch": "grotzsch",
    "groetzsch": "grotzsch",
    "myciel3": "grotzsch",  # DIMACS myciel3 = Grotzsch (11 vertices, chi = 4)
}


def list_presets() -> List[str]:
    return list(EXPECTED_CHI.keys())


def load_preset(name: str) -> Graph:
    key = name.strip().lower().replace(" ", "_")
    key = _PRESET_ALIASES.get(key, key)
    factory = _PRESETS.get(key)
    if factory is None:
        raise GraphError(
            f"Unknown preset {name!r}. Choose from: {', '.join(list_presets())}."
        )
    return factory()