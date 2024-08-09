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
from MAM_non_convex import * #MAM_non_convex import *

# parallel work parameters:
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
pool_size = comm.Get_size()

l_digit = [3]  # ,4,5,6,8]
for digit in l_digit:
    # Number of digits
    N = 10

    # List of probabilities
    with open('digit_datasets/b_centers_MNIST.pkl', 'rb') as f:  # b_centers_MNIST #b_1_mat
        l_b = pickle.load(f)
    b = l_b[digit]
    b = b[:N]
    tps = 60
    res = MAM_large_sparse_support(b, exact=False, computation_time=tps, iterations_max=500, precision=10**-6)
    # res = MAM_large_sparse_support(b, exact=False, computation_time=300, iterations_max=300,precision=10 ** -6)

    if rank == 0:
        name = 'test_False2' #inexactMAM_M10_digit3_b_300s' # f'M10_digit3_b_{tps}s'
        with open(f'{name}.pkl','wb') as f:  # local_{digit}_MAM_{pool_size}parallel_{TIME_SPENT}s_M_{len(b)}_rhos_dataset1.pkl', 'wb') as f:
            pickle.dump(res, f)

        with open(f'{name}.pkl', 'rb') as f:
            res_MAM = pickle.load(f)
        nb_pixel_side = int(res_MAM[0].shape[0]**.5)
        plt.imshow(np.reshape(res_MAM[0], (nb_pixel_side, nb_pixel_side)), cmap='hot_r')
        plt.show()

