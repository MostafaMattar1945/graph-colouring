
"""Project demo: graph presets, bounds, parsers, and Genetic Algorithm."""

from pathlib import Path

from src.graph_model import (
    EXPECTED_CHI,
    clique_lower_bound,
    greedy_upper_bound,
    list_presets,
    load_preset,
    parse_csv,
    parse_dimacs,
    parse_text,
)

# GA imports - Member 3
from src.common import GAParams
from src.ga import solve_ga


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def main() -> None:

    # ==========================================
    # PART 1 - M1: Graph Presets and Bounds
    # ==========================================

    print("\n===== PART 1: GRAPH PRESETS AND BOUNDS =====")

    print("Presets:", ", ".join(list_presets()))

    for name, chi in EXPECTED_CHI.items():
        g = load_preset(name)
        lo = clique_lower_bound(g)
        hi = greedy_upper_bound(g)

        print(
            f"  {name:16} n={len(g.vertices):2} m={len(g.edges):2}  "
            f"{lo} <= chi={chi} <= {hi}"
        )

    # ==========================================
    # PART 2 - M1: File Parsers
    # ==========================================

    print("\n===== PART 2: FILE PARSERS =====")

    txt = parse_text(
        (DATA / "exam_schedule.txt").read_text(encoding="utf-8")
    )

    csv_g = parse_csv(
        (DATA / "cycle_c5.csv").read_text(encoding="utf-8")
    )

    col = parse_dimacs(
        (DATA / "complete_k5.col").read_text(encoding="utf-8")
    )

    print(
        f"Files: exam_schedule.txt n={len(txt.vertices)}  "
        f"cycle_c5.csv n={len(csv_g.vertices)}  "
        f"complete_k5.col n={len(col.vertices)}"
    )

    # ==========================================
    # PART 3 - M3: Genetic Algorithm
    # ==========================================

    print("\n===== PART 3: GENETIC ALGORITHM =====")

    # Load a graph for the GA
    ga_graph = load_preset("exam_schedule")

    # Set GA parameters
    params = GAParams(
        pop_size=50,
        generations=200,
        crossover_rate=0.8,
        mutation_rate=0.05,
        elite_count=2,
        selection="tournament",
        tournament_size=3,
        patience=50,
        seed=42
    )

    # Run Genetic Algorithm
    result = solve_ga(ga_graph, params)

    # Display the result
    print("Algorithm:", result.algorithm)
    print("Success:", result.success)
    print("Number of colours:", result.k_used)
    print("Conflicts:", result.conflicts)
    print("Execution time:", round(result.time_ms, 2), "ms")
    print("Generations:", result.iterations)

    print("\nColouring:")

    if result.colouring is not None:
        for vertex, colour in result.colouring.items():
            print(f"  Vertex {vertex}: Colour {colour}")
    else:
        print("No valid colouring found.")

    print("\nGA Log:")

    for message in result.log:
        print(" -", message)


if __name__ == "__main__":
    main()
