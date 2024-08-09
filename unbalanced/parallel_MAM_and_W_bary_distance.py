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
# Operator splitting PARALLEL METHOD
from operator_splitting_parallel import *

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
with open('data_base_images_unbalanced_2.pkl', 'rb') as f:   #b_centers_MNIST #b_1_mat  data_base_images_unbalanced #data_base_images_unbalanced_40_40
    b = pickle.load(f)
b = b[:50]


TIME_SPENT = 1000


## Compute Wasserstein barycenter in TIME_SPENT s and store it
# with open(f'local_4_MAM_10parallel_100s_M_100_rhos_dataset1.pkl', 'rb') as f:  #centersMNIST #dataset1
#     RES_OS = pickle.load(f)
RES_OS = {}
rho = 100
# l_rho = [50]
# for rho in l_rho:
res = Operator_splitting_parallel(b, M_dist, computation_time=TIME_SPENT, iterations_min=5, iterations_max=100000, rho=rho, nb_pixel_side=nb_pixel_side)

if rank == 0:
    with open(f'local_MAM_{pool_size}parallel_{TIME_SPENT}s_M_{len(b)}_{rho}rho_data_unbalanced.pkl', 'wb') as f:
        pickle.dump(res, f)




# ## Compute Wasserstein barycenter distances and store them
# # load results
# l_para = [7] # [20, 10, 1]
# for para in l_para:
#     with open(f'MAM_random_{para}parallel_4000s_M_60_rho2000.pkl', 'rb') as f:
#         RES_OS = pickle.load(f)
#     print(f'MAM_random_{para}parallel_4000s_M_60_rho2000.pkl')
#     sys.stdout.flush()

#     rho = 2000
#     nb_random = 1
#     nb_iterations = RES_OS[rho][-1]
#     Time = RES_OS[rho][2][:nb_iterations]

#     # select the index of the result every 100 seconds:
#     cumsum_Time = np.cumsum(Time)
#     interval = 30
#     l_index = [] #[1] # iterzation 1 could have not the good properties to compute the LP and solve the Wasserstein distance
#     for i in range(len(cumsum_Time) - 1):
#         if cumsum_Time[i] % interval > interval-interval/10 and cumsum_Time[i+1] % interval < interval/10:
#             l_index.append(i)
#     # the last index is also added : this is the result after 2000 seconds
#     l_index.append(i)

#     # l_index is the list of indices I want to evaluate the barycentric distance:
#     # with open(f'barycentric_distance_OS-DR_low_memory_{TIME_SPENT}s_M_{len(b)}_rho{rho}.pkl', 'rb') as f:
#     #     barycentric_distance = pickle.load(f)
# #     barycentric_distance = {}
#     try:
#         with open(f'barycentric_distance_MAM_random_{para}parallel_4000s_M_60.pkl', 'rb') as f:
#             barycentric_distance = pickle.load(f)
#         print('data found')
#         sys.stdout.flush()
#     except:
#         print('not found')
#         barycentric_distance = {}
#     for i in l_index:
#         if cumsum_Time[i] not in barycentric_distance.keys() :
#             p = RES_OS[rho][1][:,i]
#             print(i, p.shape)
#             sys.stdout.flush()
#             p = np.abs(p) / (np.sum(np.abs(p))) ##########
#             res = barycentric_d_para(p, b, M_dist)
#             print(i, res[0])
#             sys.stdout.flush()
#             barycentric_distance[cumsum_Time[i]] = res

#             if rank == 0:
#                 with open(f'barycentric_distance_MAM_random_{para}parallel_4000s_M_60.pkl', 'wb') as f:
#                     pickle.dump(barycentric_distance, f)


            
            
            
            
            
            
            
            
            
            
            
            

## Compute Wassserstein barycnetric distance for uniform distribution
# i = 0
# p = np.ones(resolution)/resolution 
# print(i, p.shape)
# sys.stdout.flush()
# res = barycentric_d_para(p, b, M_dist)
# print(i, res[0])
# sys.stdout.flush()
# barycentric_distance[cumsum_Time[i]] = res

