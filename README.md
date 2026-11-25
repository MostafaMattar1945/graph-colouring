# Graph Colouring Solver (AI310 / CS361)

Backtracking, Genetic Algorithm and Memetic Algorithm for graph colouring (Python 3.10+).

## Status

| Part | Owner | State |
|------|-------|-------|
| Shared contract (`common.py`) and graph model (`graph_model.py`) | M1 | done |
| Backtracking solver | M2 | not in the repo yet |
| Genetic Algorithm + operators | M3 | not in the repo yet |
| Memetic Algorithm + local search | M4 | not in the repo yet |
| UI (`app.py`, Streamlit) | M5 | not in the repo yet |
| Metrics, benchmark, integration tests | M6 | not in the repo yet |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

Run everything from the project root.

```bash
python run_demo.py        # presets, bounds and file parsers sanity check
python -m pytest -q       # unit tests
# streamlit run app.py    # will work once M5 adds app.py
```

## Import rule (everyone)

The code lives in the `src/` package. Always import from `src`:

```python
from src.common import Graph, Colouring, GAParams, Result
```

Do **not** write `from common import ...`. The Team Guide PDF shows a flat layout
without `src/`, so its snippets need `src.` added. Mixing both styles creates two
different `Graph` and `GraphError` classes.

## Layout

```
src/common.py         Colouring, GAParams, Result (the Interface Contract)
src/graph_model.py    Graph, parsers, random_graph, bounds, presets
data/                 example graphs: .txt, .csv, DIMACS .col
tests/                pytest suites
run_demo.py           quick M1 check
```

## Graph API (M1)

- `Graph.from_edges(vertices, edges)`, `g.vertices`, `g.adj`, `g.edges`
- `g.degree(v)`, `g.neighbors(v)` (sorted list, use it instead of looping over `g.adj[v]`)
- `g.copy()`, `g.count_conflicts(colouring)`, `g.is_valid(colouring)`
- Edges are undirected, duplicates are ignored, self-loops raise `GraphError`.

Loading graphs:

- `parse_text`: one edge per line (`A B`; `A-B` also works); `#` starts a comment.
- `parse_csv`: `u,v` per line, header optional.
- `parse_dimacs`: `p edge n m` then `e u v` (1-based ids).
- `load_preset(name)`: `exam_schedule`, `petersen`, `cycle_c5`, `complete_k5`,
  `bipartite_k33`, `grotzsch` (alias `myciel3`).
- `random_graph(n, p, seed)`, `greedy_upper_bound(g)`, `clique_lower_bound(g)`.

## Things to know

- `count_conflicts` ignores vertices that have no colour yet. `is_valid` also fails
  on an incomplete colouring. Solvers and tests must check both `success` and
  `conflicts` (never report `success=True` when `conflicts > 0`).
- Vertex names: `random_graph` uses `"0".."n-1"`, DIMACS files use `"1".."n"`.
- `clique_lower_bound` is a heuristic clique, not the maximum one. On C5, Petersen
  and Grotzsch it returns 2 while the chromatic number is larger. That is still a
  valid lower bound.