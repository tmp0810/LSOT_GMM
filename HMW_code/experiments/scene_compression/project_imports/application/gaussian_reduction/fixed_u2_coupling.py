"""Materialize the current fixed-U2 scalar method's implicit mean coupling.

Reuse its representation, H6 orders, root-priced NW kernel and spherical
certificate unchanged.  The only extension is retaining each NW plan with
mass divided by H.  Repeated COO pairs are additive, not candidate edges to
optimize.  The geometric-center scalar is diagnostic: it never rescales mass.
"""

from time import perf_counter
from types import MappingProxyType

import numpy as np

from application.common import SparseCoupling
from application.gaussian_reduction.main_coupling import (
    BITS, H, MainCouplingSolver, _as_core_mixture, _validate_application_mixture,
)
from experiments.nc_hot_bw import prepare_compact_q2, prepare_projected_orders
from experiments.nc_hot_bw.candidate_graph_ot import (
    MeanRootGaugeResult,
    _direct_nw_root_gauge_transport_kernel,
    spherical_anchor_from_root_gauge_trusted,
)
from experiments.nc_hot_bw.representation import sha256_array


METHOD = "fixed_u2_h6_mean_root_coupling"


class FixedU2MeanCouplingSolver(MainCouplingSolver):
    """No calibration, spatial-only order, candidate optimization or T4.

The inherited initialization uses equal-mass quantiles in the first fixed-q2
order.  The immutable source representation/order cache is also inherited.
"""

    def __init__(self):
        super().__init__(refinement_iterations=0)

    def __call__(self, source, target):
        started = perf_counter()
        _validate_application_mixture(source, name="source")
        _validate_application_mixture(target, name="target")
        hit = self.prepare_source(source)
        cached = self._prepared_source
        pa = cached.orders.mixture
        target_started = perf_counter()
        pb = prepare_compact_q2(
            _as_core_mixture(target), self.compiled_q2, fast_d3_roots=True
        )
        represented = perf_counter()
        ordered_b = prepare_projected_orders(pb, self.projection_bank, bits=BITS)
        ordered = perf_counter()

        fields = [[], [], [], []]
        objectives, counts, matched = [], [], []
        for oa, ob in zip(cached.orders.orders, ordered_b.orders):
            values = _direct_nw_root_gauge_transport_kernel(
                pa.weights, oa, pa.mixture.means, pa.roots,
                pb.weights, ob, pb.mixture.means, pb.roots,
            )
            rows, cols, masses, costs, count, objective = values
            objectives.append(objective)
            counts.append(count)
            matched.append(float(np.sum(masses[:count])))
            for destination, array in zip(fields, (rows, cols, masses, costs)):
                destination.append(array[:count])
        rows, cols, masses, costs = [np.concatenate(parts) for parts in fields]
        masses /= H
        materialized = perf_counter()

        row_sum = np.bincount(rows, weights=masses, minlength=source.support_size)
        col_sum = np.bincount(cols, weights=masses, minlength=target.support_size)
        residual = max(float(np.max(np.abs(row_sum - source.weights))),
                       float(np.max(np.abs(col_sum - target.weights))))
        if residual > 1e-9 or not np.all(np.isfinite(costs)):
            raise RuntimeError("fixed-U2 mean coupling failed feasibility/price audit")
        upper = MeanRootGaugeResult(
            squared_objective=float(np.mean(objectives)),
            per_order_squared_objectives=np.asarray(objectives),
            per_order_edge_counts=np.asarray(counts, dtype=np.int64),
            per_order_matched_masses=np.asarray(matched), h=H,
            source_support=source.support_size, target_support=target.support_size,
            evaluated_edge_visits=int(sum(counts)),
        )
        score = spherical_anchor_from_root_gauge_trusted(pa, pb, upper)
        if not np.isclose(np.dot(masses, costs), upper.squared_objective,
                          rtol=2e-12, atol=2e-14):
            raise RuntimeError("materialized plan differs from scalar upper price")
        rows = np.ascontiguousarray(source.original_indices[rows])
        cols = np.ascontiguousarray(target.original_indices[cols])
        for array in (rows, cols, masses, costs):
            array.setflags(write=False)
        diagnostics = {
            "method": METHOD, "method_role": "fixed_u2_main_mean_coupling",
            "q": 2, "h": H, "view_dimension": 2, "hilbert_bits": BITS,
            "u2_sha256": sha256_array(self.compiled_q2.u2),
            "order_bank_sha256": sha256_array(self.projection_bank.directions),
            "ordering_ground": "fixed_radial_haar_phi2",
            "cost_semantics": "squared_principal_root_gauge",
            "costs_are_exact_gaussian_bw": False,
            "objective_is_exact_mixture_wasserstein": False,
            "candidate_graph_constructed": False, "refinement_performed": False,
            "requested_refinement_iterations": 0, "effective_refinement_iterations": 0,
            "coo_duplicates_are_additive": True, "coo_entries": len(masses),
            "coupling_semantics": "arithmetic_mean_of_six_original_nw_plans",
            "coupling_objective_is_scalar_geometric_center": False,
            "root_upper_bound": score.upper_bound,
            "spherical_lower_bound": score.lower_bound,
            "spherical_scalar_distance": score.distance,
            "source_cache_hit": hit,
            "source_cached_representation_ms": cached.representation_ms,
            "source_cached_order_ms": cached.order_ms,
            "target_representation_ms": (represented - target_started) * 1000,
            "target_order_ms": (ordered - represented) * 1000,
            "nw_root_price_and_output_ms": (materialized - ordered) * 1000,
            "audit_certificate_and_host_map_ms": (perf_counter() - materialized) * 1000,
            "online_total_ms": (perf_counter() - started) * 1000,
            "marginal_residual": residual,
        }
        return SparseCoupling(
            rows=rows, columns=cols, masses=masses, costs=costs,
            source_indices=source.original_indices, target_indices=target.original_indices,
            source_weights=source.weights, target_weights=target.weights,
            objective=upper.squared_objective, marginal_residual=residual,
            diagnostics=MappingProxyType(diagnostics),
        )
