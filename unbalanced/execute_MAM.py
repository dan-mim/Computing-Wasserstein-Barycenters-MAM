# -*- coding: utf-8 -*-
"""
@author: Daniel Mimouni
"""

# %% Imports
# Basics
import matplotlib.pyplot as plt
import pickle
import numpy as np
# CPU management:
from mpi4py import MPI
import sys

# My codes:
# mam (balanced)
from mam import MAM

# parallel work parameters:
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
pool_size = comm.Get_size()


# List of probabilities
with open('dataset_letters/data_letters_mAM_lower.pkl', 'rb') as f:
    b = pickle.load(f)
M = 10
b = b[:M]

# compute the result for the problem
res = MAM(b, exact=False,
          rho=1000,
          gamma=1,
          keep_track=True,
          name=f'unbalanced.pkl',
          visualize=True, evry_it=10,
          computation_time=1000, iterations_max=12000,
          precision=10 ** -20)
