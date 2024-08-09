import matplotlib.pyplot as plt
import numpy.matlib
import pickle
import numpy as np
# CPU management:
from mpi4py import MPI
import sys
from LP_Wasserstein_distance import *
from MAM_non_convex import *
from Sinkhorn_distance import *


def bary_distance_sink(p, b, Mat_dist, splitting_work, rank):
    """
    * p is the computed barycenter
    * b is a collection of measures
    * Mat_dist is a dict with the distance matrices
    The function compute the barycenter distance between p and the b[i]'s
    """
    # compute Wasserstein distances
    distance = 0
    # p
    p = p/np.sum(p)
    J = p > 10**-6
    p = p[J]
    p = p/np.sum(p)
    for m in splitting_work[rank]:
        # q
        q = b[m]
        q = q/np.sum(q)
        I = q>0
        q = q[I]
        q = q/np.sum(q)
        # Compute distance
        WD = sinkhorn_descent(p, q, Mat_dist[m][J,:], lambda_sinkhorn=1, iterations=10) # Wasserstein_distance_LP(p, q, Mat_dist[m][J,:])
        print(f'WD is computed for {m}, in {WD[-1]}s and WD = {WD[0]}')
        sys.stdout.flush()
        distance = distance + WD[0]
    # Gather and get the sum of the distances:
    l_distance = comm.gather(distance, root=0)
    W_distance = 0
    if rank == 0:
        W_distance = np.sum(l_distance) / M
        print(f'The barycentric distance is {W_distance}')
        sys.stdout.flush()
    return(W_distance)


# List of probabilities
with open('dataset/dataset_altschuler.pkl', 'rb') as f:
    b = pickle.load(f)
M = 10
b = b[:M]


# Barycenter with MAM
with open('outputs/res_MAM_14400s_10ellipses.pkl', 'rb') as f:
    res = pickle.load(f)
p = res[0] #res[1][:,9] # res[0]
p = p/np.sum(p)

# Barycenter with Altschuler
with open('dataset/ours_exact_just_computed_ellipses.pkl', 'rb') as f:
    resA = pickle.load(f)
R = 60 * 10 - 10 + 1
pA, xedges, yedges = np.histogram2d(resA[:, 0] * R, resA[:, 1] * R, bins=R, range=[[0, R], [0, R]], weights=resA[:,2])
pA = pA/np.sum(pA)
pA = np.reshape(pA, (pA.size,))


# Compute distance matrix
# Parallelization initialization:
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
pool_size = comm.Get_size()
splitting_work = division_tasks(M, pool_size)
R = p.shape[0]
S = b[0].shape[0]
Mat_dist = build_M_dist(rank, splitting_work, pool_size, comm, R, S, b)

bary_dist_pA = bary_distance_sink(pA, b, Mat_dist, splitting_work, rank)
if rank == 0:
    print('bary dist Altschuler = ', bary_dist_pA)
bary_dist_p = bary_distance_sink(p, b, Mat_dist, splitting_work, rank)
if rank == 0:
    print('bary dist MAM = ', bary_dist_p)
bary_dist_p10 = bary_distance_sink(res[1][:,10], b, Mat_dist, splitting_work, rank)
if rank == 0:
    print('bary dist MAM 10 iterations = ', bary_dist_p10)


# # Compute the barycenter distance for the times in l_tps
# len_ = 40
# l_tps = np.linspace(0, 240, len_)
# # list of times
# Time_MAM = res[2]
# CumsumTime_MAM = np.cumsum(Time_MAM)/60
# l_i_MAM = [min(enumerate(CumsumTime_MAM), key=lambda x: abs(x[1]-tps))[0] for tps in l_tps]
#
# if rank == 0:
#     l_bary_dist = []
# for i_t in l_i_MAM[1:]:
#     bary_dist = bary_distance_sink(res[1][:,i_t], b, Mat_dist, splitting_work, rank)
#     if rank == 0:
#         print('bary dist= ', bary_dist)
#         l_bary_dist.append(l_bary_dist)
# if rank == 0:
#     name = f'l_bary_dist_every_{len_}min'
#     with open(f'outputs/{name}.pkl','wb') as f:
#         pickle.dump(l_bary_dist, f)
#
# if rank == 0:
#     name = f'l_bary_dist_every_{len_}min'
#     with open(f'outputs/{name}.pkl', 'rb') as f:
#         l_bary_dist = pickle.load(f)
#     liste_tps = np.linspace(0, 240, len_)
#     plt.figure()
#     plt.plot(liste_tps[1:], np.array(l_bary_dist))
#     plt.show()