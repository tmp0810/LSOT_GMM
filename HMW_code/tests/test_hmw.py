import ast
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hmw import GaussianMixture, OfficialHCP, hmw, prepare_cholesky
from hmw.representation import normalize_weights


class TestBackend:
    """Deterministic fixture, explicitly not a Hilbert implementation."""
    def order(self, x):
        return np.lexsort(tuple(x[:,i] for i in range(x.shape[1]-1, -1, -1)))

    def nw(self, a, b, oa, ob):
        aa, bb = a[oa].copy(), b[ob].copy()
        i = j = 0
        rows, cols, mass = [], [], []
        while i < len(aa) and j < len(bb):
            v = min(aa[i], bb[j])
            rows.append(oa[i]); cols.append(ob[j]); mass.append(v)
            aa[i] -= v; bb[j] -= v
            if aa[i] <= 0:
                i += 1
            if bb[j] <= 0:
                j += 1
        return np.array(rows), np.array(cols), np.array(mass)


def source(n, d, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n,d,d))
    return GaussianMixture(rng.normal(size=(n,d)), z @ z.transpose(0,2,1) + .25*np.eye(d), rng.dirichlet(np.full(n,2.)))


def original_functions():
    # Execute only declared functions/classes from the recorded core; avoid
    # importing the unrelated legacy q=2/Sinkhorn package as a side effect.
    norm = ROOT / "experiments/_original/nc_hot_bw/representation.py"
    tree = ast.parse(norm.read_text())
    ns = {"np": np}
    node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == "normalize_weights")
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(norm), "exec"), ns)
    original = ROOT / "experiments/_original/cholesky_official_hmw.py"
    tree = ast.parse(original.read_text())
    keep = [n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))]
    env = {"np": np, "dataclass": dataclass, "perf_counter": perf_counter,
           "rep": SimpleNamespace(normalize_weights=ns["normalize_weights"]),
           "OfficialHCP": OfficialHCP, "official_order_checked": lambda b,x:b.order(x)}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(original), "exec"), env)
    return env


class HMWTests(unittest.TestCase):
    def test_representation_original_parity(self):
        old = original_functions()
        for d in (1,2,3,5,20):
            a = source(13,d,77+d)
            before = [x.copy() for x in (a.means,a.covariances,a.weights)]
            x,w = prepare_cholesky(a)
            xx,ww = old["prepare_cholesky"](a)
            np.testing.assert_array_equal(x,xx)
            np.testing.assert_array_equal(w,ww)
            self.assertEqual(x.shape, (13,d+d*(d+1)//2))
            for original, current in zip(before,(a.means,a.covariances,a.weights)):
                np.testing.assert_array_equal(original,current)

    def test_pipeline_parity_and_marginals(self):
        old = original_functions()
        for d in (1,3,5):
            a,b = source(17,d,30+d),source(11,d,60+d)
            out = hmw(a,b,TestBackend())
            prior = old["cholesky_official_hmw"](a,b,TestBackend())
            self.assertEqual(out.squared_cost,prior.squared_cost)
            for key in ("rows","cols","mass"):
                np.testing.assert_array_equal(getattr(out,key),getattr(prior,key))
            np.testing.assert_allclose(np.bincount(out.rows,weights=out.mass,minlength=17),normalize_weights(a.weights),atol=1e-14)
            np.testing.assert_allclose(np.bincount(out.cols,weights=out.mass,minlength=11),normalize_weights(b.weights),atol=1e-14)
            self.assertLessEqual(len(out.mass),27)

    def test_invalid_weights(self):
        for weights in ([0,1],[-1,2],[np.nan,1],[np.inf,1],[]):
            with self.assertRaises(ValueError):
                normalize_weights(weights)

    def test_spd_not_silently_repaired(self):
        for cov in (np.diag([1.,-1.]), np.array([[1.,.2],[0.,1.]])):
            with self.assertRaises((ValueError,np.linalg.LinAlgError)):
                prepare_cholesky(GaussianMixture(np.zeros((1,2)),cov[None],np.ones(1)))

    @unittest.skipUnless(os.environ.get("HCP_EXTENSION"), "set HCP_EXTENSION for actual compiled C++ parity")
    def test_native_parity(self):
        backend = OfficialHCP(os.environ["HCP_EXTENSION"])
        old = original_functions()
        for d in (1,3,5,10):
            a,b = source(31,d,100+d),source(22,d,200+d)
            out = hmw(a,b,backend)
            prior = old["cholesky_official_hmw"](a,b,backend)
            self.assertEqual(out.squared_cost,prior.squared_cost)
            np.testing.assert_array_equal(out.mass,prior.mass)


if __name__ == "__main__":
    unittest.main()
