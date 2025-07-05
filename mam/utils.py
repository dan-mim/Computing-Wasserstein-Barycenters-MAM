# Imports
import numpy as np
import scipy
from scipy import spatial
import time

# Vectorize function that project vectors onto a simplex, inspired by https://gist.github.com/mblondel/c99e575a5207c76a99d714e8c6e08e89
def projection_simplex(V, z=1, axis=None):
    """
    Projection of x onto the simplex, scaled by z:
        P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2
    z: float or array
        If array, len(z) must be compatible with V
    axis: None or int
        axis=None: project V by P(V.ravel(); z)
        axis=1: project each V[i] by P(V[i]; z[i])
        axis=0: project each V[:, j] by P(V[:, j]; z[j])
    This code is inspired by Condat work on the simple

    (c) Daniel Mimouni
    """
    if axis == 1:
        n_features = V.shape[1]
        U = np.sort(V, axis=1)[:, ::-1]
        z = np.ones(len(V)) * z
        cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
        ind = np.arange(n_features) + 1
        cond = U - cssv / ind > 0
        rho = np.count_nonzero(cond, axis=1)
        theta = cssv[np.arange(len(V)), rho - 1] / rho
        return np.maximum(V - theta[:, np.newaxis], 0)

    elif axis == 0:
        return projection_simplex(V.T, z, axis=1).T

    else:
        V = V.ravel().reshape(1, -1)
        return projection_simplex(V, z, axis=1).ravel()


# Computing the distance matrix:
def distance_matrix(M, S, exact=False):
    """
    This function compute the distance matrix for the *exact* (free support) problem
    or the *inexact* problem (fixed support)
    M is the number of considered densities
    S is the size of the probability densities support
    """
    if not exact:
        R = S   # R is the size of the barycenter support we look for
        # eps_R and eps_S are the discretization size of the barycenter image (for R) and the sample images (for S)
        eps_R = 1 / R ** .5
        eps_S = 1 / S ** .5

        la = np.linspace(0, R-1, R)
        lb = np.linspace(0, S-1, S)

        x_R, y_R = la % R ** .5 * eps_R + eps_R / 2, la // R ** .5 * eps_R + eps_R / 2
        x_S, y_S = lb % S ** .5 * eps_S + eps_S / 2, lb // S ** .5 * eps_S + eps_S / 2

        XA = np.array([x_R, y_R]).T
        XB = np.array([x_S, y_S]).T
        M_dist = spatial.distance.cdist(XA, XB, metric='euclidean')

    if exact:
        # Method that respect Bogward theorems:
        K = int(np.round(S ** .5))
        X, Y = np.meshgrid(np.append(np.arange(0, K-1, 1 / M), K-1), np.append(np.arange(0, K-1, 1 / M), K-1))
        ptsK = np.column_stack((X.ravel(), Y.ravel()))

        X, Y = np.meshgrid(np.linspace(0,K-1,K), np.linspace(0,K-1,K))
        ptsk = np.column_stack((X.ravel(), Y.ravel()))

        # Calcul de la distance
        M_dist = spatial.distance.cdist(ptsK, ptsk)

    return(M_dist**2)


# This function compute the exact Wasserstein distance between two probability distributions
# It runs the massive linear program with the HiGHS method
def Wasserstein_distance_LP(p, q, M_dist):
    """
    Inputs:
    *p: (n x 1) probability measure
    *q: (m x 1) probability measure
    *M_dist: (n x m) matrix giving distances between the locations of supports p and q

    Outputs:
    *Distance_exact: (float) Wasserstein distance between p and q
    *Pi_exact: (n x m) Transport matrix between p and q
    *Time: (float) time spent to compute the result

    Infos:
    LP resolution solving: Wasserstein_distance(p,q) (p and q have the same support)
    *function to minimize: c @ x
    *constraints: A @ x = p and B @ x = q
    The massive linear program is run with the HiGHS method (see more on HiGHS.dev)
    """
    # Time management:
    start = time.time()

    # Dimensions
    R_, S_ = M_dist.shape

    assert len(q) == S_, 'la taille de q ne concorde pas'
    assert len(p) == R_, 'la taille de p ne concorde pas'

    # Distance matrix as a vector c.T (1,R_*S_)
    c = np.reshape(M_dist, (R_ * S_,))

    # Building A: Constraint (right sum of Pi to get p)
    shape_A = (R_, R_ * S_)
    indices_values_A = np.ones(R_ * S_)
    # lines indices: time_spent = 8s for R_=S_=1600
    line_indices_A = np.array([])
    for r in range(R_):
        line_indices_A = np.append(line_indices_A, np.ones(S_) * r)
    # column indices: time_spent < 0.1s for R_=S_=1600
    column_indices_A = np.linspace(0, R_ * S_ - 1, R_ * S_)

    # Building B: Constraint (left sum of Pi to get q)
    shape_B = (S_, R_ * S_)
    indices_values_B = np.ones(R_ * S_)
    # lines indices: time_spent for R_=S_=1600
    line_indices_B = np.array([])
    for s in range(S_):
        line_indices_B = np.append(line_indices_B, np.ones(R_) * s)
    # column indices: time_spent for R_=S_=1600
    column_indices_B = np.array([])
    for s in range(S_):
        column_indices_B = np.append(column_indices_B, np.linspace(0, S_ * (R_ - 1), R_) + s)

    # Building A_eq = concatenate((A, B), axis=0)
    shape_Aeq = (R_ + S_, R_ * S_)
    indices_values_Aeq = np.append(indices_values_A, indices_values_B)
    line_indices_Aeq = np.append(line_indices_A, line_indices_B + R_)
    column_indices_Aeq = np.append(column_indices_A, column_indices_B)
    A_eq = csc_matrix((indices_values_Aeq, (line_indices_Aeq, column_indices_Aeq)), shape_Aeq)

    # b_eq:
    b_eq = np.append(p, q)

    # Resolution:
    res = scipy.optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, method='highs')

    # computing time:
    end = time.time()
    Time = np.round((end - start), 2)

    # Output:
    try:
        # Optimization terminated successfully.
        Distance_exact = res.fun
        Pi_exact = res.x
        Pi_exact = np.reshape(Pi_exact, (R_, S_))

        return (Distance_exact, Pi_exact, Time)

    except:
        # The problem is infeasible.
        message = res.message
        return (message, [], Time)

