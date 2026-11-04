"""The existing four projection laws, with streamed lifting and true-W2 selection."""
from dataclasses import fields
import torch

from lsot.projections import project_gaussians, sample_projection_bank
from ..common import tensor_digest, timed
from ..geometry import GaussianCost
from ..plans import AveragePlan, LiftedPlan, SortedProjection
from . import SolverResult


def bank_slice(bank, start, stop):
    return type(bank)(*(getattr(bank, f.name)[start:stop] for f in fields(bank)))


def project_chunked(gmm, bank, kind, chunk=4096):
    # Bound the eigensolver batch even for Mix; B itself also chunks pair eigensolves.
    result = gmm.means.new_empty((gmm.count, bank.count))
    for k in range(0, gmm.count, chunk):
        result[k:k+chunk] = project_gaussians(gmm.means[k:k+chunk],
                                             gmm.covariances[k:k+chunk], bank, kind)
    return result


class LSOTSolver:
    def __init__(self, source, method, options, timings, bank=None):
        mode, kind = method.split("-", 1)
        if mode not in ("avg", "min") or kind not in ("Mix", "SMix", "B", "B1D"):
            raise ValueError(f"Unknown LSOT method: {method}")
        self.source, self.mode, self.kind = source, mode, kind
        self.options, self.timings, self.device = options, timings, source.means.device
        count = options.get("projections", 100)
        maximum = options.get("bank_size", 200)
        if not 1 <= count <= maximum:
            raise ValueError("Require 1 <= projections <= bank_size for nested banks")
        with timed(timings, "bank_seconds", self.device):
            self.bank = bank if bank is not None else sample_projection_bank(
                3, maximum, kind=kind, seed=options.get("projection_seed", 0),
                dtype=source.means.dtype, device=self.device).prefix(count)
            self.bank_hash = tensor_digest(*(getattr(self.bank, f.name) for f in fields(self.bank)))
        with timed(timings, "source_geometry_seconds", self.device):
            self.cost = GaussianCost(source, options.get("pair_chunk", 4096))
        self.x = []
        with timed(timings, "source_projection_sort_seconds", self.device):
            batch = options.get("projection_batch", 8)
            for start in range(0, self.bank.count, batch):
                bank_part = bank_slice(self.bank, start, min(start+batch, self.bank.count))
                values = project_chunked(source, bank_part, kind, options.get("component_chunk", 4096))
                self.x.extend(SortedProjection(values[:, ell], source.weights)
                              for ell in range(bank_part.count))

    def solve(self, target):
        costs, plans, best, selected = [], [], None, None
        for start in range(0, self.bank.count, self.options.get("projection_batch", 8)):
            part = bank_slice(self.bank, start, min(start+self.options.get("projection_batch", 8), self.bank.count))
            with timed(self.timings, "target_projection_sort_seconds", self.device):
                y = project_chunked(target, part, self.kind, self.options.get("component_chunk", 4096))
                ys = [SortedProjection(y[:, ell], target.weights) for ell in range(part.count)]
            with timed(self.timings, "lift_and_full_cost_seconds", self.device):
                for ell, sorted_y in enumerate(ys):
                    plan = LiftedPlan(self.x[start+ell], sorted_y)
                    cost = self.cost.plan_cost(target, plan)
                    costs.append(cost)
                    if best is None or cost < costs[selected]:
                        best, selected = plan, start + ell
                    if self.mode == "avg":
                        plans.append(plan)
        plan = best if self.mode == "min" else AveragePlan(plans)
        with timed(self.timings, "marginal_audit_seconds", self.device):
            error = plan.marginal_error(self.source.weights, target.weights)
        if error > self.options.get("marginal_tolerance", 1e-8):
            raise RuntimeError(f"Lifted plan marginal error {error}")
        return SolverResult(plan, costs[selected] if self.mode == "min" else sum(costs)/len(costs),
            {"projection_index": selected if self.mode == "min" else None,
             "projection_costs": costs, "bank_sha256": self.bank_hash,
             "marginal_l1": error, "average_storage": "implicit weighted sum" if self.mode == "avg" else None})
