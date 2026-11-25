"""Quick M1 sanity check: presets, bounds, and file parsers."""

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

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def main() -> None:
    print("Presets:", ", ".join(list_presets()))
    for name, chi in EXPECTED_CHI.items():
        g = load_preset(name)
        lo, hi = clique_lower_bound(g), greedy_upper_bound(g)
        print(
            f"  {name:16} n={len(g.vertices):2} m={len(g.edges):2}  "
            f"{lo} <= chi={chi} <= {hi}"
        )

    txt = parse_text((DATA / "exam_schedule.txt").read_text(encoding="utf-8"))
    csv_g = parse_csv((DATA / "cycle_c5.csv").read_text(encoding="utf-8"))
    col = parse_dimacs((DATA / "complete_k5.col").read_text(encoding="utf-8"))
    print(
        f"Files: exam_schedule.txt n={len(txt.vertices)}  "
        f"cycle_c5.csv n={len(csv_g.vertices)}  "
        f"complete_k5.col n={len(col.vertices)}"
    )


if __name__ == "__main__":
    main()
