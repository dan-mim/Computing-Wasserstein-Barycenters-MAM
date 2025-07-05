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
with open('dataset/dataset_altschuler.pkl', 'rb') as f:
    b = pickle.load(f)
M = 10
b = b[:M]
for m in range(M):
    b[m] = b[m] /np.sum(b[m])

# compute the result for the problem
tps = 24 * 3600
res = MAM(b, exact=True,
          rho=4000,
          keep_track=True,
          name=f'outputs_MAM/saving_MAM_evry10i_3.pkl',
          visualize=True, evry_it=10,
          computation_time=tps, iterations_max=12000,
          precision=10 ** -20)

if rank == 0:
    name = f'res_MAM_{tps}s_{M}ellipses'
    with open(f'outputs_MAM/{name}.pkl','wb') as f:
        pickle.dump(res, f)

    with open(f'outputs_MAM/{name}.pkl', 'rb') as f:
        res_MAM = pickle.load(f)
    print(res[-1])
    sys.stdout.flush()
    nb_pixel_side = int(res_MAM[0].shape[0]**.5)
    plt.imshow(np.reshape(res_MAM[0], (nb_pixel_side, nb_pixel_side)), cmap='hot_r')
    plt.show()

