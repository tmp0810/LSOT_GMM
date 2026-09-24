"""Averaged and minimum lifted sliced transport for Gaussian mixtures."""
from .gaussians import GMM
from .plans import MinimumLSOTResult, SparsePlan, average_lsot, minimum_lsot, solve_mw2
from .maps import BarycentricMap

__all__ = ["GMM", "SparsePlan", "MinimumLSOTResult", "average_lsot", "minimum_lsot", "solve_mw2", "BarycentricMap"]
