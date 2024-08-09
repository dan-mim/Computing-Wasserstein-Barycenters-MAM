# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 12:43:38 2023

@author: mimounid
"""

# %% Imports

# Basics
import numpy as np
import pandas as pd
import scipy.io
import time
import random
import matplotlib.pyplot as plt
import numpy.matlib
import pickle
import numpy as np
# CPU management:
from mpi4py import MPI
import sys

# My codes:
# MAM (balanced)
from MAM_non_convex import *

# # Exact method that solves the LP ant find the barycentric measure
# from resolution_LP_barycenter import *
# # Cuturi PARALLEL METHOD barycenter
# from Cuturi_parallel import *
# # Operator splitting developped method
# from operator_splitting_low_memory import *
# # Operator splitting PARALLEL METHOD
# from operator_splitting_parallel import *
# # Operator splitting PARALLEL METHOD with RANDOM selection
# from operator_splitting_parallel_random import *
# # Operator splitting PARALLEL METHOD with iterative improvement: HEURISTIC METHOD
# from operator_splitting_parallel_heuristic import *
# # Iterative Bregman Projections method
# from barycenter_IBP import *
# # Exact method that solves the LP and find the Wasserstein distance between two measures PARALLELIZED
# from Wasserstein_dist_parallel import *

# Distance matrix

# Picture resolution
nb_pixel_side = 40
resolution = nb_pixel_side * nb_pixel_side

# ## Get M the squared Euclidean distance Matrix between all 40x40 pixels
# # To make sure I don't mess up indexing things, I'll set up a list of locations and reshape it into a matrix
# # So when I calculate a pairwise distance in the matrix, I can easily associate it to the location in the vector
# locations_vec = np.array(range(resolution))
# locations_arr = np.reshape(locations_vec, (nb_pixel_side, nb_pixel_side))
#
# M_dist = np.zeros((resolution, resolution))
# # Having 4 "for" loops is a bit embarassing, but I'm not trying to think too hard right now
# for i1 in range(nb_pixel_side):
#     for j1 in range(nb_pixel_side):
#         for i2 in range(nb_pixel_side):
#             for j2 in range(nb_pixel_side):
#                 M_dist[locations_arr[i1, j1], locations_arr[i2, j2]] = ((i1 - i2) ** 2 + (j1 - j2) ** 2)  # np.sqrt

## Get M the squared Euclidean distance Matrix between all the pixels for an exat measure
with open(f'M_dist_5.pkl', 'rb') as f:
    M_dist = pickle.load(f)

# parallel work parameters:
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
pool_size = comm.Get_size()

l_digit = [3] #,4,5,6,8]
for digit in l_digit:
    # Number of digits
    N = 5

    # List of probabilities
    with open('digit_datasets/b_1_mat.pkl', 'rb') as f:   #b_centers_MNIST #b_1_mat
        l_b = pickle.load(f)
    b = l_b[digit]
    b = b[:N]


    res_MAM = {}
    res = MAM_large_sparse_support(b, exact=False, name=f'outputs/res_ellipses3D.pkl', exact=False, computation_time=tps, iterations_max=10000, precision=10 ** -6)
    res_MAM = res

    if rank == 0:
        with open(f'test_exact.pkl', 'wb') as f: #local_{digit}_MAM_{pool_size}parallel_{TIME_SPENT}s_M_{len(b)}_rhos_dataset1.pkl', 'wb') as f:
            pickle.dump(res_MAM, f)

nb_pixel_side = int(M_dist.shape[0]**.5)
with open(f'test_exact.pkl', 'rb') as f:
    res_MAM = pickle.load(f)
plt.imshow(np.reshape(res_MAM[0], (nb_pixel_side,nb_pixel_side)) , cmap='hot_r')
plt.show()


            
            
            
            
            
            
            
            
            
            
            
            

## Compute Wassserstein barycnetric distance for uniform distribution
# i = 0
# p = np.ones(resolution)/resolution 
# print(i, p.shape)
# sys.stdout.flush()
# res = barycentric_d_para(p, b, M_dist)
# print(i, res[0])
# sys.stdout.flush()
# barycentric_distance[cumsum_Time[i]] = res

