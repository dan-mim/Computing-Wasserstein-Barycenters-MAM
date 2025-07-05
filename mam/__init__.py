"""
Method of Averaged Marginals (mam)

A package to compute Wasserstein barycenters using operator splitting
and parallel marginal projections. Based on:

Mimouni et al., "Computing Wasserstein Barycenter via Operator Splitting:
The Method of Averaged Marginals", SIAM J. Math. Data Sci., 2024

05-07-2025

@author: mimounid
"""

from .solver import MAM, build_M_dist, division_tasks
from .utils import distance_matrix, projection_simplex, Wasserstein_distance_LP

