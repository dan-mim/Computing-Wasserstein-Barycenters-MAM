# -*- coding: utf-8 -*-
"""
Inspired from Operator_splitting_low_memory but with a arallel dimension

@author: mimounid
"""

# Imports

# Basics
import numpy as np
import time
# Import parallel work management:
from mpi4py import MPI
import sys


def division_tasks(nb_tasks, pool_size):
    """
    Inputs: (int)
    *nb_tasks
    *pool_size : number of CPU/GPU to divide the tasks between

    Outputs:
    rearranged: numpy list of lists so that rearranged[i] should be treated by CPU[i] (rank=i)
    """
    # The tasks can be equaly divided for each CPUs
    if nb_tasks % pool_size == 0:
        rearranged = np.array([i for i in range(nb_tasks)])
        rearranged = np.split(rearranged, pool_size)

    # Some CPUs will receive more tasks
    else:
        div = nb_tasks // pool_size
        congru = nb_tasks % pool_size
        rearranged1 = np.array([i for i in range(div * congru + congru)])
        rearranged1 = np.split(rearranged1, congru)
        rearranged2 = np.array([i for i in range(div * congru + congru, nb_tasks)])
        rearranged2 = np.split(rearranged2, pool_size - congru)
        rearranged = rearranged1 + rearranged2

    # Output:
    return (rearranged)

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


# Optimized Operator Splitting method
def MAM_general(b, M_dist, computation_time=500, iterations_min=100, iterations_max=100000, rho=1000,
                                  nb_pixel_side=40, gamma=0):
    """
    Input:
    *b: (resolution x n) vector collection of n probability distribution 
    *M_dist:( resolution x resolution) is the distance matrix of every pixels 
    *rho: (float) hyperparameter responsible for the algorithm speed to converge to a solution. Set to 1000
    if rho is precised in the arguments of the function then this rho will be used. 
    *nb_pixel_side: (float) is pixel number of the side of the square picture
    *gamma: (float) unbalanced parameter -> if gamma != 0 then the algorithm goes directly in the unbalanced configuration

    Output:
    *p: (resolution x 1) the probability distribution of the calculated barycenter
    *Time: execution time of each iteration
    *theta
    *iterations_k: (int) number of iterations 

    Infos:
    This function use less momory storage than a direct vectorized implementation
    This function compute the barycenter of probability distribution in the case of a support identical for all distribution and for the targeted barycenter
    If we need different support, the function can be easily modified in order to change the matrix D.

    (c) Daniel Mimouni 2023
    """""

    st1 = time.time()

    # Parallelization initialization:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    pool_size = comm.Get_size()

    # Parameter initializations
    # Number of probability distributions
    M_ = len(b)

    # Dimension of the support of the targeted barycenter
    resolution = nb_pixel_side * nb_pixel_side
    R = resolution

    # M_dist get normalized:
    max_D = np.max(M_dist)
    M_dist = M_dist / max_D

    # storage of the transport plans
    theta = {}
    sum_theta_mean = 0
    inv_sum_S = 0
    S = []
    for m in range(M_):
        # dimensionality reduction
        I = b[m] > 0
        n_not_null = np.sum(I)
        S.append(n_not_null)
        inv_sum_S = inv_sum_S + 1 / n_not_null

        # Stored in a dictionnary
        theta[m] = -1 / rho * M_dist[:, I] + np.ones(n_not_null) / R

        # to compute p
        theta_mean_1 = np.mean(theta[m], axis=1)
        sum_theta_mean = sum_theta_mean + theta_mean_1

    # probability:
    p = sum_theta_mean / inv_sum_S

    # Keep track:
    Time = np.zeros(iterations_max + 1)  # time per iteration



    # Algorithm iterations:
    spent_time = 0
    iterations_k = 0
    splitting_work = division_tasks(M_, pool_size)
    while (iterations_k < iterations_min) or (spent_time < computation_time and iterations_k < iterations_max):
        iterations_k = iterations_k + 1
        start = time.time()

        # Initialize
        sum_theta_mean_local = np.zeros(R)

        # Unbalanced Wasserstein barycenter -> Computing coefficient t_unbalanced:
        t_unbalanced = 1                        # if balanced Wasserstein barycenter
        if gamma != 0 and iterations_k > 1 :    # if unbalanced
            # Compute t_unbalanced
            coeff_unbalanced_local = 0
            for m in splitting_work[rank]:
                coeff_unbalanced_local = coeff_unbalanced_local + np.linalg.norm(np.sum(theta[m], axis=1) - p )**2 / S[m]
            # Gather
            coeff_unbalanced = comm.gather(coeff_unbalanced_local, root=0)
            coeff_unbalanced = np.array(coeff_unbalanced)
            coeff_unbalanced = np.sum(coeff_unbalanced, axis=0)
            # Bcast:
            coeff_unbalanced = comm.bcast(coeff_unbalanced, root=0)

            # Define unbalanced coefficient
            t_unbalanced = min(1, gamma / rho / coeff_unbalanced**.5)

            # if rank==0:
            #     print(t_unbalanced)
            #     sys.stdout.flush()


        # PARALLELIZATION
        # iterate over probabilities
        for m in splitting_work[rank]:
            # index of M_dist I use
            I = b[m] > 0

            # Get the new theta[m]
            # deltaU
            deltaU = np.sum(theta[m], axis=1) / S[m] - p / S[m]
            # W for the projection
            theta[m] = theta[m] - 1 / rho * M_dist[:, I] - 2 * t_unbalanced * np.expand_dims(deltaU, axis=1)
            # W get normalized before its projection onto the simplex
            theta[m] = theta[m] / b[m][I]
            # the transport plan is un-normalized after the projection onto the simplex
            theta[m] = projection_simplex(theta[m], z=1, axis=0) * b[m][I]
            theta[m] = theta[m] + t_unbalanced * np.expand_dims(deltaU, axis=1)

            # mean of theta:
            theta_mean_1 = np.mean(theta[m], axis=1)
            sum_theta_mean_local = sum_theta_mean_local + theta_mean_1

        # Gather and bcast the local solutions
        # Gather and get the sum of the theta_mean's:
        l_sum_theta_mean = comm.gather(sum_theta_mean_local, root=0)
        l_sum_theta_mean = np.array(l_sum_theta_mean)
        sum_theta_mean = np.sum(l_sum_theta_mean, axis=0)
        # Bcast:
        sum_theta_mean = comm.bcast(sum_theta_mean, root=0)

        # probability:
        p = sum_theta_mean / inv_sum_S


        # Time management:
        end = time.time()
        time_spent = np.round((end - start), 2)
        Time[iterations_k] = time_spent

        # manage time at a global scale:
        spent_time = time.time() - st1
        spent_time = comm.bcast(spent_time , root=0)

    end = time.time()
    time_spent = np.round((end - st1) / 60, 2)
#     print(f"rank = {rank} : The {iterations_k} iterations took {time_spent} minutes")
#     sys.stdout.flush()
    
    # Output
    return (p, Time, time_spent, iterations_k)








