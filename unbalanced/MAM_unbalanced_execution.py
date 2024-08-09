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
# MAM parallel with the unbalanced configuration if needed
from MAM_parallel_unbalanced import *

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

# List of probabilities ######################## pkl ad b[:50] !!!!!!!!!!!! ##########
with open('data_base_images_unbalanced_2.pkl',
          'rb') as f:  # b_centers_MNIST #b_1_mat  data_base_images_unbalanced #data_base_images_unbalanced_40_40 #data_base_images_unbalanced_2 #data_base_images_unbalanced_weights
    b = pickle.load(f)
b = b[:50]
# b = b[3][:60]
######################## pkl ad b[:50] !!!!!!!!!!!! ##########

TIME_SPENT = 1000

rho = 100  #best rho try smaller and then longer time
l_gamma = [10**-3, 1, 10, 50, 70, 200, 300, 500, 1000] # [1000, 300, 200, 100, 70, 50, 10, 1, 10**-3] # [10**-4, 10**-3, 10**-2, 10**-1, 1, 10] #, 0]
# l_rho = [50]
for gamma in l_gamma:
    RES_MAM = MAM_general(b, M_dist, computation_time=TIME_SPENT, iterations_min=5, iterations_max=100000,
                                      rho=rho, nb_pixel_side=nb_pixel_side, gamma=gamma)

    if rank == 0:
        with open(f'local_MAM_{pool_size}parallel_{TIME_SPENT}s_M_{len(b)}_{rho}rho_{gamma}unbalanced_2.pkl', 'wb') as f:
            pickle.dump(RES_MAM, f)

    RES_MAM = {}
