# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 12:43:38 2023

@author: mimounid
"""

# %% Imports

# Basics
import pickle
# CPU management:
from mpi4py import MPI
import sys

# My codes:
# mam parallel with the unbalanced configuration if needed
from MAM_parallel_double_unbalanced import *

# Exact method that solves the LP and find the Wasserstein distance between two measures PARALLELIZED
# from Wasserstein_dist_parallel import *

# Distance matrix

# Picture resolution
nb_pixel_side = 80
resolution = nb_pixel_side * nb_pixel_side

## Get M the squared Euclidean distance Matrix between all 40x40 pixels
# To make sure I don't mess up indexing things, I'll set up a list of locations and reshape it into a matrix
# So when I calculate a pairwise distance in the matrix, I can easily associate it to the location in the vector
locations_vec = np.array(range(resolution))
locations_arr = np.reshape(locations_vec, (nb_pixel_side, nb_pixel_side))

M_dist = np.zeros((resolution, resolution))
# Having 4 "for" loops is a bit embarassing, but I'm not trying to think too hard right now
for i1 in range(nb_pixel_side):
    for j1 in range(nb_pixel_side):
        for i2 in range(nb_pixel_side):
            for j2 in range(nb_pixel_side):
                M_dist[locations_arr[i1, j1], locations_arr[i2, j2]] = ((i1 - i2) ** 2 + (j1 - j2) ** 2)  # np.sqrt

# parallel work parameters:
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
pool_size = comm.Get_size()

# List of probabilities
with open('data_base_images_unbalanced_weights.pkl', 'rb') as f:  # b_centers_MNIST #b_1_mat
    #data_base_images_unbalanced_2 #data_base_images_unbalanced_weights
    b = pickle.load(f)
b = [b[3], b[4], b[5]]

TIME_SPENT = 100

rho = 100  #best rho try smaller and then longer time

l_gamma = [1000] #, 300, 70, 10**-3]
# l_gamma = l_gamma[::-1] #
l_eta = [10**-4, 10**-3, 2*10**-3, 4*10**-3, 5*10**-3 , 6*10**-3 , 1, 10, 1000]
# l_eta = l_eta[::-1]

gamma_rho = []
for g in l_gamma:
    for e in l_eta:
        gamma_rho.append((g,e))
# gamma_rho = [(1000, 0), (1000, 0.2), (1000, 0.5), (1000, 0.8), (1000,8), (300, 0.2), (300, 0.5), (300, 0.8), (300, 8), (70,8), (10**-3, 0.2), (10**-3, 0.5), (10**-3, 0.8), (10**-3, 1), (10**-3, 2), (10**-3, 6), (10**-3, 8)]
for j in range(len(gamma_rho)):
    RES_MAM = {}
    gamma, eta = gamma_rho[j]
    RES_MAM = MAM_double_general(b, M_dist, computation_time=TIME_SPENT, iterations_min=5, iterations_max=100000,
                                      rho=rho, nb_pixel_side=nb_pixel_side, gamma=gamma, eta=eta)

    if rank == 0:
        with open(f'local_MAM_{pool_size}parallel_{TIME_SPENT}s_M_{len(b)}_{rho}rho_{gamma}__{eta}unbalanced_weights.pkl', 'wb') as f:
            pickle.dump(RES_MAM, f)

    RES_MAM = {}


# eta = 1000
# RES_MAM = MAM_double_general(b, M_dist, computation_time=1000, iterations_min=5, iterations_max=100000,
#                                   rho=rho, nb_pixel_side=nb_pixel_side, gamma=1000, eta=eta)
#
# if rank == 0:
#     with open(f'local_MAM_{pool_size}parallel_{TIME_SPENT}s_M_{len(b)}_{rho}rho_{1000}-{eta}unbalanced_2.pkl', 'wb') as f:
#         pickle.dump(RES_MAM, f)