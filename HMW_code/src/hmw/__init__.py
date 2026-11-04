"""Small API for the measured Cholesky + official HCP + NW pipeline.

This is the historical Figure 3 path, not the newer canonical implementation
described elsewhere in the manuscript. See docs/IMPLEMENTATION_STATUS.md.
"""
from .representation import GaussianMixture, prepare_cholesky
from .hilbert import OfficialHCP, official_order_checked
from .transport import HMWResult, hmw

__all__ = ["GaussianMixture", "prepare_cholesky", "OfficialHCP",
           "official_order_checked", "HMWResult", "hmw"]
