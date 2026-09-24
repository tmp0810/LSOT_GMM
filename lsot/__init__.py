"""Averaged lifted sliced transport for finite Gaussian mixtures."""
from .gaussians import GMM
from .plans import SparsePlan, average_lsot, solve_mw2
from .maps import BarycentricMap

__all__ = ["GMM", "SparsePlan", "average_lsot", "solve_mw2", "BarycentricMap"]
