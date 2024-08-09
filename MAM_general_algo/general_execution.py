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
# MAM
from MAM import *


# parallel work parameters:
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
pool_size = comm.Get_size()

# List of probabilities
N = 100
with open('digit_datasets/b_1_mat.pkl', 'rb') as f:   #b_centers_MNIST #b_1_mat
    l_b = pickle.load(f)
b = l_b[3]
b = b[:N]


res = MAM(b, exact=False, name=f'outputs/res.pkl', computation_time=500, iterations_max=10000, precision=10 ** -6)


if rank == 0:
    with open(f'test_exact.pkl', 'wb') as f:
        pickle.dump(res, f)



            
            
            
            
            
            
            
            
            
            
            
            

## Compute Wassserstein barycnetric distance for uniform distribution
# i = 0
# p = np.ones(resolution)/resolution 
# print(i, p.shape)
# sys.stdout.flush()
# res = barycentric_d_para(p, b, M_dist)
# print(i, res[0])
# sys.stdout.flush()
# barycentric_distance[cumsum_Time[i]] = res

