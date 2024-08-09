import numpy as np
import time
import matplotlib.pyplot as plt
# Import parallel work management:
from mpi4py import MPI
import sys
import pickle
import scipy
from MAM_non_convex import *

# PARAMETERS
M = 100
# with open('digit_datasets/b_centers_MNIST.pkl', 'rb') as f:  # b_centers_MNIST #b_1_mat
#     l_b = pickle.load(f)
# b = l_b[3]
# b = b[:M]
pool_size = 4

# what lacks:
D = 50000
e = 500
rho = 50
b = {}
theta = {}
Mat_dist = {}
S = {}
for m in range(M):
    theta[m] = np.random.rand(D,e)
    Mat_dist[m] = np.random.rand(D,e)
    S[m] = e
    b[m] = np.random.rand(e)
p = np.random.rand(D)
rank = 2
# PARALLELIZATION
sum_theta_mean_local = np.sum(theta[m], axis=1)
# iterate over probabilities
splitting_work = division_tasks(M, pool_size)
for m in splitting_work[rank]:
    start = time.time()
    # index of M_dist I use
    I = b[m] > 0

    # Get the new theta[m]
    # deltaU
    deltaU = np.sum(theta[m], axis=1) / S[m] - p / S[m]
    # W for the projection
    theta[m] = theta[m] - 1 / rho * Mat_dist[m] - 2 * np.expand_dims(deltaU, axis=1)
    # W get normalized before its projection onto the simplex
    theta[m] = theta[m] / b[m][I]
    # the transport plan is un-normalized after the projection onto the simplex
    theta[m] = projection_simplex(theta[m], z=1, axis=0) * b[m][I]

    # Pi[m] = theta[m].copy()

    theta[m] = theta[m] + np.expand_dims(deltaU, axis=1)

    # mean of theta:
    theta_mean_1 = np.mean(theta[m], axis=1)
    sum_theta_mean_local = sum_theta_mean_local + theta_mean_1
    print(time.time() - start)

# method matricielle
M = M//pool_size
THETA = np.random.rand(D,e*M)
MAT = np.random.rand(D,e*M)
B = np.random.rand(e*M)


start = time.time()
# Get the new theta[m]
# deltaU
deltaU = np.sum(THETA, axis=1) / S[m] - p / S[m]
# W for the projection
THETA = THETA - 1 / rho * MAT - 2 * np.expand_dims(deltaU, axis=1)
# W get normalized before its projection onto the simplex
THETA = THETA / B
# the transport plan is un-normalized after the projection onto the simplex
THETA = projection_simplex(THETA, z=1, axis=0) * B

# Pi[m] = theta[m].copy()

THETA = THETA + np.expand_dims(deltaU, axis=1)

# mean of theta:
theta_mean_1 = np.mean(THETA, axis=1)
sum_theta_mean_local = sum_theta_mean_local + theta_mean_1
print(time.time() - start)
