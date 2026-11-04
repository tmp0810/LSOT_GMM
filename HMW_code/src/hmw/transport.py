"""Stage 3: ordered mass matching and full-feature coupling price."""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from .representation import prepare_cholesky


@dataclass(frozen=True)
class HMWResult:
    distance: float
    squared_cost: float
    rows: np.ndarray
    cols: np.ndarray
    mass: np.ndarray
    stages_ms: dict


def hmw(source, target, backend):
    """Compute one-order Cholesky HMW. Indices refer to the supplied atom lists."""
    started = tick = perf_counter()
    xa, wa = prepare_cholesky(source)
    xb, wb = prepare_cholesky(target)
    if xa.shape[1] != xb.shape[1]:
        raise ValueError("source and target dimensions differ")
    stages = {"representation": (perf_counter() - tick) * 1000}
    tick = perf_counter()
    oa, ob = backend.order(xa), backend.order(xb)
    stages["official_order"] = (perf_counter() - tick) * 1000
    tick = perf_counter()
    rows, cols, mass = backend.nw(wa, wb, oa, ob)
    stages["official_nw"] = (perf_counter() - tick) * 1000
    tick = perf_counter()
    delta = xa[rows] - xb[cols]
    squared = float(mass @ np.einsum("ij,ij->i", delta, delta))
    if not np.isfinite(squared) or squared < -1e-12:
        raise RuntimeError("invalid HMW cost")
    squared = max(squared, 0.0)
    stages["feature_price"] = (perf_counter() - tick) * 1000
    stages["total"] = (perf_counter() - started) * 1000
    return HMWResult(float(np.sqrt(squared)), squared, rows, cols, mass, stages)
