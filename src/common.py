from dataclasses import dataclass, field
from typing import Dict, List, Optional

Colouring = Dict[str, int]


@dataclass
class GAParams:
    pop_size: int = 50
    generations: int = 200
    crossover_rate: float = 0.8
    mutation_rate: float = 0.05
    elite_count: int = 2
    selection: str = "tournament"
    tournament_size: int = 3
    patience: int = 50
    ls_iterations: int = 20
    ls_rate: float = 1.0
    seed: Optional[int] = None


@dataclass
class Result:
    algorithm: str
    success: bool
    colouring: Optional[Colouring]
    k_used: Optional[int]
    conflicts: int
    time_ms: float
    iterations: int
    backtracks: int = 0
    history: List[int] = field(default_factory=list)
    log: List[str] = field(default_factory=list)
    timed_out: bool = False


from .graph_model import Graph  # noqa: E402,F401