"""Paper-faithful q=2 Hilbert candidate graph and restricted-flow backend."""

from __future__ import annotations

import hashlib
from time import perf_counter
from types import MappingProxyType
from typing import Any

import numpy as np

from experiments.nc_hot_bw import (
    Mixture,
    build_candidate_graph,
    compile_q2_map,
    exact_bw_edge_costs_eigh,
    fixed_orthogonal_bank,
    prepare_compact_q2,
    prepare_orders,
    refine_candidate_graph_with_info,
)

from .calibration import u2_sha256
from .contracts import (
    ApplicationMixture,
    CandidateProblem,
    SparseCoupling,
    as_application_mixture,
)


def _bank_sha256(bank: tuple[np.ndarray, ...]) -> str:
    stacked = np.ascontiguousarray(np.stack(bank), dtype=np.float64)
    return hashlib.sha256(stacked.tobytes()).hexdigest()


def _freeze(array: np.ndarray, dtype: Any) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(array, dtype=dtype).copy())
    result.setflags(write=False)
    return result


class CandidateBackend:
    """Reusable application boundary around the three NC-HOT-BW stages.

    The backend deliberately uses full ``D=d+3`` orthogonal Hilbert views and
    exact host edge prices.  It does not call the package-level ``nc_hot_bw``
    alias, whose current fast H9 implementation has different semantics.
    """

    def __init__(self, u2: np.ndarray, *, h: int = 8, bits: int = 20) -> None:
        if isinstance(h, bool) or int(h) != h or h < 1:
            raise ValueError("h must be a positive integer")
        if isinstance(bits, bool) or int(bits) != bits or bits < 1 or bits > 62:
            raise ValueError("bits must be an integer in [1, 62]")
        self.compiled_q2 = compile_q2_map(np.asarray(u2, dtype=np.float64))
        self.h = int(h)
        self.bits = int(bits)
        self.order_bank = fixed_orthogonal_bank(
            self.compiled_q2.feature_dimension, self.h
        )
        self.u2_sha256 = u2_sha256(self.compiled_q2.u2)
        self.order_bank_sha256 = _bank_sha256(self.order_bank)

    def build_problem(
        self,
        source: ApplicationMixture,
        target: ApplicationMixture,
    ) -> CandidateProblem:
        if source.ambient_dimension != self.compiled_q2.ambient_dimension:
            raise ValueError("source dimension does not match U2")
        if target.ambient_dimension != source.ambient_dimension:
            raise ValueError("source and target ambient dimensions must match")

        started = perf_counter()
        prepared_source = prepare_compact_q2(
            Mixture(source.means, source.covariances, source.weights),
            self.compiled_q2,
        )
        prepared_target = prepare_compact_q2(
            Mixture(target.means, target.covariances, target.weights),
            self.compiled_q2,
        )
        represented = perf_counter()
        orders_source = prepare_orders(
            prepared_source, self.order_bank, self.h, bits=self.bits
        )
        orders_target = prepare_orders(
            prepared_target, self.order_bank, self.h, bits=self.bits
        )
        ordered = perf_counter()
        graph = build_candidate_graph(
            prepared_source.weights,
            orders_source.orders,
            prepared_target.weights,
            orders_target.orders,
        )
        graphed = perf_counter()
        timing = MappingProxyType(
            {
                "representation": 1.0e3 * (represented - started),
                "orders": 1.0e3 * (ordered - represented),
                "candidate_union": 1.0e3 * (graphed - ordered),
                "support_total": 1.0e3 * (graphed - started),
            }
        )
        diagnostics = MappingProxyType(
            {
                "source_components": source.support_size,
                "target_components": target.support_size,
                "raw_edges": int(graph.raw_edges),
                "union_edges": int(len(graph.rows)),
                "graph_components": int(graph.components),
                "cycle_rank": int(graph.cycle_rank),
                "source_max_collision_fraction": float(
                    orders_source.maximum_collision_fraction
                ),
                "target_max_collision_fraction": float(
                    orders_target.maximum_collision_fraction
                ),
                "method": "q2_full_feature_hilbert_exact_host_cost",
            }
        )
        return CandidateProblem(
            source=source,
            target=target,
            prepared_source=prepared_source,
            prepared_target=prepared_target,
            graph=graph,
            h=self.h,
            bits=self.bits,
            u2_sha256=self.u2_sha256,
            order_bank_sha256=self.order_bank_sha256,
            timing_ms=timing,
            diagnostics=diagnostics,
        )

    def build_problem_from_arrays(
        self,
        source_means: Any,
        source_covariances: Any,
        source_weights: Any,
        target_means: Any,
        target_covariances: Any,
        target_weights: Any,
    ) -> CandidateProblem:
        return self.build_problem(
            as_application_mixture(
                source_means, source_covariances, source_weights
            ),
            as_application_mixture(
                target_means, target_covariances, target_weights
            ),
        )

    @staticmethod
    def exact_bw_costs(problem: CandidateProblem) -> np.ndarray:
        """Evaluate exact Gaussian BW squared costs on candidate edges only."""

        return exact_bw_edge_costs_eigh(
            problem.prepared_source,
            problem.prepared_target,
            problem.rows,
            problem.columns,
        )

    @staticmethod
    def solve(
        problem: CandidateProblem,
        edge_costs: np.ndarray,
        *,
        max_refinement_iterations: int | None = None,
    ) -> SparseCoupling:
        """Solve exact or budgeted restricted balanced transport.

        ``None`` requests exact HiGHS optimization on the candidate graph.  A
        finite value is explicitly a budgeted-refinement ablation.
        """

        costs = np.ascontiguousarray(np.asarray(edge_costs, dtype=np.float64))
        if costs.shape != problem.rows.shape:
            raise ValueError("edge_costs must align with candidate graph edges")
        if not bool(np.all(np.isfinite(costs))):
            raise ValueError("edge_costs must be finite")
        # An edge-independent offset leaves a unit-mass balanced LP unchanged
        # and lets host reduced costs be negative without violating the core's
        # numerical expectations.  We report the original-cost objective.
        offset = min(0.0, float(np.min(costs)))
        solver_costs = np.ascontiguousarray(costs - offset, dtype=np.float64)
        started = perf_counter()
        result = refine_candidate_graph_with_info(
            problem.graph,
            solver_costs,
            problem.prepared_source.weights,
            problem.prepared_target.weights,
            max_refinement_iterations=max_refinement_iterations,
        )
        solved = perf_counter()
        masses = np.asarray(result.masses, dtype=np.float64)
        if bool(np.any(masses < -3.0e-12)):
            raise RuntimeError("restricted flow returned negative mass")
        masses = np.maximum(masses, 0.0)
        row_sum = np.bincount(
            problem.rows,
            weights=masses,
            minlength=problem.source.support_size,
        )
        column_sum = np.bincount(
            problem.columns,
            weights=masses,
            minlength=problem.target.support_size,
        )
        residual = max(
            float(np.max(np.abs(row_sum - problem.source.weights))),
            float(np.max(np.abs(column_sum - problem.target.weights))),
        )
        if residual > 3.0e-8:
            raise RuntimeError(
                f"restricted coupling marginal residual {residual:.3e} exceeds 3e-8"
            )
        mapped_rows = problem.source.original_indices[problem.rows]
        mapped_columns = problem.target.original_indices[problem.columns]
        objective = float(np.dot(masses, costs))
        diagnostics = dict(problem.diagnostics)
        diagnostics.update(
            {
                "solver_mode": result.mode,
                "solver_optimal": bool(result.optimal),
                "solver_budget_exhausted": bool(result.budget_exhausted),
                "solver_iterations": int(result.iterations_completed),
                "selected_initial_order": int(result.selected_initial_order),
                "active_edges": int(np.count_nonzero(masses > 0.0)),
                "marginal_residual": float(residual),
                "objective": objective,
                "cost_offset_for_solver": float(offset),
                "solve_ms": 1.0e3 * (solved - started),
                "u2_sha256": problem.u2_sha256,
                "order_bank_sha256": problem.order_bank_sha256,
                **{f"support_{key}_ms": value for key, value in problem.timing_ms.items()},
            }
        )
        return SparseCoupling(
            rows=_freeze(mapped_rows, np.int64),
            columns=_freeze(mapped_columns, np.int64),
            masses=_freeze(masses, np.float64),
            costs=_freeze(costs, np.float64),
            source_indices=problem.source.original_indices,
            target_indices=problem.target.original_indices,
            source_weights=problem.source.weights,
            target_weights=problem.target.weights,
            objective=objective,
            marginal_residual=float(residual),
            diagnostics=MappingProxyType(diagnostics),
        )

    def solve_exact_bw(
        self,
        source: ApplicationMixture,
        target: ApplicationMixture,
        *,
        max_refinement_iterations: int | None = None,
    ) -> SparseCoupling:
        problem = self.build_problem(source, target)
        pricing_started = perf_counter()
        costs = self.exact_bw_costs(problem)
        pricing_ms = 1.0e3 * (perf_counter() - pricing_started)
        coupling = self.solve(
            problem,
            costs,
            max_refinement_iterations=max_refinement_iterations,
        )
        diagnostics = dict(coupling.diagnostics)
        diagnostics["pricing_ms"] = pricing_ms
        return SparseCoupling(
            rows=coupling.rows,
            columns=coupling.columns,
            masses=coupling.masses,
            costs=coupling.costs,
            source_indices=coupling.source_indices,
            target_indices=coupling.target_indices,
            source_weights=coupling.source_weights,
            target_weights=coupling.target_weights,
            objective=coupling.objective,
            marginal_residual=coupling.marginal_residual,
            diagnostics=MappingProxyType(diagnostics),
        )
