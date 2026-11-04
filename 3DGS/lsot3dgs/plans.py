"""Streaming balanced plans, including proportional lifts of tied fibers."""
import torch


class Plan:
    def marginals(self, chunk=65536):
        a = self.prototype.new_zeros(self.shape[0])
        b = self.prototype.new_zeros(self.shape[1])
        for i, j, p in self.edges(chunk):
            a.index_add_(0, i, p)
            b.index_add_(0, j, p)
        return a, b

    def marginal_error(self, a, b):
        aa, bb = self.marginals()
        return max(float((aa-a).abs().sum()), float((bb-b).abs().sum()))

    def sparse(self, max_edges=2_000_000):
        indices, masses, count = [], [], 0
        for i, j, p in self.edges(65536):
            count += len(p)
            if count > max_edges:
                raise MemoryError("Diagnostic sparse materialization exceeds max_edges")
            indices.append(torch.stack((i, j)))
            masses.append(p)
        return torch.sparse_coo_tensor(torch.cat(indices, 1), torch.cat(masses), self.shape,
                                       check_invariants=True).coalesce()

    def dense(self, max_entries=1_000_000):
        if self.shape[0] * self.shape[1] > max_entries:
            raise MemoryError("Dense plan is for small validation cases only")
        return self.sparse(max_edges=max_entries * 2).to_dense()


class EdgePlan(Plan):
    def __init__(self, rows, cols, mass, shape):
        if len(rows) != len(cols) or len(rows) != len(mass) or not len(mass):
            raise ValueError("Malformed edge plan")
        if not bool(torch.isfinite(mass).all()) or bool((mass <= 0).any()):
            raise ValueError("Edges must have finite positive mass")
        self.rows, self.cols, self.mass, self.shape = rows, cols, mass, tuple(shape)
        self.prototype = mass

    def edges(self, chunk=65536):
        for k in range(0, len(self.mass), chunk):
            yield self.rows[k:k+chunk], self.cols[k:k+chunk], self.mass[k:k+chunk]


def monotone(a, b):
    ca, cb = a.cumsum(0), b.cumsum(0)
    ca = torch.cat((ca[:-1].clamp(max=1), ca.new_ones(1)))
    cb = torch.cat((cb[:-1].clamp(max=1), cb.new_ones(1)))
    cuts = torch.cat((a.new_zeros(1), ca, cb)).sort().values
    left, mass = cuts[:-1], cuts[1:] - cuts[:-1]
    keep = mass > 0
    return (torch.searchsorted(ca, left[keep], right=True).clamp(max=len(a)-1),
            torch.searchsorted(cb, left[keep], right=True).clamp(max=len(b)-1), mass[keep])


class SortedProjection:
    def __init__(self, values, weights):
        self.values, self.order = torch.sort(values, stable=True)
        self.weights = weights[self.order]
        _, self.groups, self.counts = torch.unique_consecutive(
            self.values, return_inverse=True, return_counts=True)
        self.starts = self.counts.cumsum(0) - self.counts
        self.group_mass = weights.new_zeros(len(self.counts)).index_add_(0, self.groups, self.weights)


class LiftedPlan(Plan):
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.shape = (len(x.values), len(y.values))
        self.prototype = x.weights
        self.gi, self.gj, self.mass = monotone(x.group_mass, y.group_mass)
        ordinary = (x.counts[self.gi] == 1) & (y.counts[self.gj] == 1)
        self.ordinary = torch.where(ordinary)[0]
        self.tied = torch.where(~ordinary)[0].tolist()
        self.edge_count = int((x.counts[self.gi] * y.counts[self.gj]).sum())

    def edges(self, chunk=65536):
        x, y = self.x, self.y
        for ids in self.ordinary.split(chunk):
            yield x.order[x.starts[self.gi[ids]]], y.order[y.starts[self.gj[ids]]], self.mass[ids]
        # A large fiber may be dense. Stream its Cartesian product exactly.
        for k in self.tied:
            g, h = int(self.gi[k]), int(self.gj[k])
            nx, ny = int(x.counts[g]), int(y.counts[h])
            for start in range(0, nx * ny, chunk):
                ids = torch.arange(start, min(start+chunk, nx*ny), device=self.prototype.device)
                u, v = x.starts[g] + ids // ny, y.starts[h] + ids % ny
                p = self.mass[k] * (x.weights[u] / x.group_mass[g]) * (y.weights[v] / y.group_mass[h])
                yield x.order[u], y.order[v], p


class AveragePlan(Plan):
    def __init__(self, plans):
        if not plans:
            raise ValueError("Cannot average an empty bank")
        self.plans = tuple(plans)
        self.shape, self.prototype = plans[0].shape, plans[0].prototype

    def edges(self, chunk=65536):
        for plan in self.plans:
            for i, j, p in plan.edges(chunk):
                yield i, j, p / len(self.plans)


def plan_change(a, b, max_edges=2_000_000):
    """Exact L1 for single plans; a rigorous L1 upper bound for equal-bank averages.

    Return None if a pathological tie exceeds the diagnostic memory budget.
    Such a run may reach its iteration cap but must not claim convergence.
    """
    if isinstance(a, AveragePlan) and isinstance(b, AveragePlan):
        if len(a.plans) != len(b.plans):
            return None
        values = [plan_change(x, y, max_edges) for x, y in zip(a.plans, b.plans)]
        return None if any(x is None for x in values) else sum(values) / len(values)
    try:
        difference = (a.sparse(max_edges) - b.sparse(max_edges)).coalesce()
        return float(difference.values().abs().sum())
    except MemoryError:
        return None
