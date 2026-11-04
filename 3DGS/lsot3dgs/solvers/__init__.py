from dataclasses import dataclass


@dataclass
class SolverResult:
    plan: object
    objective: float
    diagnostics: dict


def make_solver(method, source, options, timings):
    if method == "mw2":
        from .mw2 import MW2Solver
        return MW2Solver(source, options, timings)
    from .lsot import LSOTSolver
    return LSOTSolver(source, method, options, timings)
