"""Exact balanced POT EMD with chunked GPU Gaussian costs and a dual audit."""
import numpy as np
import torch

from ..common import timed
from ..geometry import GaussianCost
from ..plans import EdgePlan
from . import SolverResult


class BudgetExceeded(RuntimeError):
    pass


class MW2Solver:
    def __init__(self, source, options, timings):
        self.source, self.options, self.timings = source, options, timings
        self.device = source.means.device
        with timed(timings, "source_geometry_seconds", self.device):
            self.cost = GaussianCost(source, options.get("pair_chunk", 4096))

    def solve(self, target):
        import ot
        n, m = self.source.count, target.count
        # This is a conservative allocation guard, not measured peak RSS.
        estimate = 4 * n * m * 8 + 64 * (n + m)
        budget = self.options.get("max_dense_bytes", 16 * 1024**3)
        if estimate > budget:
            raise BudgetExceeded(f"MW2 estimated dense workspace {estimate} exceeds budget {budget}")
        matrix = np.empty((n, m), dtype=np.float64)
        flat = matrix.reshape(-1)
        with timed(self.timings, "dense_cost_and_transfer_seconds", self.device):
            for k in range(0, n*m, self.cost.chunk):
                ids = torch.arange(k, min(k+self.cost.chunk, n*m), device=self.device)
                flat[k:k+self.cost.chunk] = self.cost.pairs(target, ids // m, ids % m).cpu().numpy()
        with timed(self.timings, "cpu_emd_seconds"):
            scale = max(float(matrix.max()), np.finfo(np.float64).tiny)
            matrix /= scale
            a = np.ascontiguousarray(self.source.weights.cpu().numpy(), dtype=np.float64)
            b = np.ascontiguousarray(target.weights.cpu().numpy(), dtype=np.float64)
            p, info = ot.emd(a, b, matrix, numItermax=self.options.get("emd_iterations", 1_000_000),
                             numThreads=self.options.get("emd_threads", 1), log=True)
            if info.get("warning") is not None or info.get("result_code", 1) != 1:
                raise RuntimeError(f"Exact EMD did not converge: {info.get('warning')}")
            primal = float(np.einsum("ij,ij->", p, matrix))
            dual = float(np.dot(a, info["u"]) + np.dot(b, info["v"]))
            violation = 0.0
            row_chunk = max(1, min(n, 1_000_000 // m))
            for k in range(0, n, row_chunk):
                violation = max(violation, float(np.max(
                    info["u"][k:k+row_chunk, None] + info["v"][None, :] - matrix[k:k+row_chunk])))
            marginal = max(float(abs(p.sum(1)-a).sum()), float(abs(p.sum(0)-b).sum()))
            gap = abs(primal-dual)
            tol = self.options.get("emd_tolerance", 1e-8)
            if not np.isfinite(p).all() or p.min() < 0 or max(marginal, violation, gap) > tol:
                raise RuntimeError(f"EMD certificate failed: marginal={marginal}, dual={violation}, gap={gap}")
            i, j = np.nonzero(p > 0)
            masses = p[i, j].copy()
            del p, matrix
        with timed(self.timings, "plan_transfer_seconds", self.device):
            plan = EdgePlan(torch.as_tensor(i, device=self.device), torch.as_tensor(j, device=self.device),
                            torch.as_tensor(masses, device=self.device, dtype=self.source.weights.dtype), (n, m))
        return SolverResult(plan, primal * scale, {"marginal_l1": marginal,
            "dual_violation_scaled": violation, "duality_gap_scaled": gap,
            "cost_scale": scale, "dense_workspace_estimate_bytes": estimate,
            "solver": "POT exact EMD on CPU; Gaussian cost on requested device"})
