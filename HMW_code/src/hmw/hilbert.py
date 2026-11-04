"""Stage 2: adapter to the unchanged upstream median-recursive HCP sorter."""
import importlib.util
from pathlib import Path
import numpy as np

OFFICIAL_COMMIT = "930ecc98d2a6c6e884d96015919fed7de7c664f0"
SOURCE_SHA256 = "8f39fe76d99be69459b80612b5c6adb64176da6c9059d678a948b355cef0cddd"


class OfficialHCP:
    def __init__(self, extension_path):
        path = Path(extension_path).resolve(strict=True)
        spec = importlib.util.spec_from_file_location("base", str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load official HCP extension: {path}")
        self.native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.native)

    def order(self, points):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or min(points.shape) < 1:
            raise ValueError("require a nonempty n-by-D array")
        if not np.isfinite(points).all() or len(points) > np.iinfo(np.int32).max:
            raise ValueError("nonfinite features or integer index overflow")
        result = np.asarray(self.native.hilbert_order(points), dtype=np.int64)
        if result.shape != (len(points),) or not np.array_equal(np.sort(result), np.arange(len(points))):
            raise RuntimeError("official sorter did not return a permutation")
        return result

    @staticmethod
    def weights(value):
        value = np.asarray(value, dtype=np.float64)
        if value.ndim != 1 or not len(value) or not np.isfinite(value).all() or np.any(value <= 0):
            raise ValueError("require strictly positive finite weights")
        if abs(float(value.sum()) - 1.0) > 1e-12:
            raise ValueError("normalize weights before native NW")
        return value

    def nw(self, source_weights, target_weights, source_order, target_order):
        a, b = self.weights(source_weights), self.weights(target_weights)
        oa, ob = np.asarray(source_order), np.asarray(target_order)
        for order, n in ((oa, len(a)), (ob, len(b))):
            if not np.issubdtype(order.dtype, np.integer) or not np.array_equal(np.sort(order), np.arange(n)):
                raise ValueError("invalid permutation")
        raw = np.asarray(self.native.general_Plan(a[oa], b[ob]))
        if raw.ndim != 2 or raw.shape[0] != 3 or not np.isfinite(raw).all():
            raise RuntimeError("invalid native plan layout")
        mass = raw[0]
        if np.min(mass) < -1e-12 or not np.array_equal(raw[1:], raw[1:].astype(np.int64)):
            raise RuntimeError("invalid mass/indices returned by native NW")
        i, j = raw[1].astype(np.int64), raw[2].astype(np.int64)
        if np.any(i < 0) or np.any(i >= len(a)) or np.any(j < 0) or np.any(j >= len(b)):
            raise RuntimeError("native NW indices out of bounds")
        rows, cols = oa[i], ob[j]
        residual = max(np.max(abs(np.bincount(rows, weights=mass, minlength=len(a)) - a)),
                       np.max(abs(np.bincount(cols, weights=mass, minlength=len(b)) - b)))
        if residual > 1e-11:
            raise RuntimeError(f"NW marginal residual: {residual}")
        return rows, cols, mass  # Preserve exact-zero edges like the measured path.


def official_order_checked(backend, points):
    return backend.order(points)
