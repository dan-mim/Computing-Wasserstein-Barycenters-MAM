import matplotlib.pyplot as plt
import numpy.matlib
import pickle
import numpy as np
import sys
from scipy.io import savemat

with open(f'outputs/saving_MAM_evry10i_3.pkl', 'rb') as f:
    res = pickle.load(f)
P = res[1]
p = res[0]
p = p/np.sum(p)
plt.figure()
R = 60 * 10 - 10 + 1
# R = 60
plt.imshow(np.reshape(p,(R,R)), cmap='hot_r')
plt.colorbar()
plt.show()
p = np.reshape(p, (p.size,))

ExactWBD = [  0.2831, 0.2749, 0.2714, 0.2694, 0.2683,  0.2677,  0.2674, 0.2672, 0.2671,  0.2670, 0.2670, 0.2669, 0.2669, 0.2669,  0.2669, 0.2669, 0.2669, 0.2668,  0.2668, 0.2668, 0.2668, 0.2668, 0.2668, 0.2668, 0.2668, 0.2668, 0.2668,  0.2668, 0.2668, 0.2668, 0.2668, 0.2668, 0.2668, 0.2668, 0.2668, 0.2668, 0.2668,    0.2668,    0.2668,    0.2668, 0.2668, 0.2668, 0.2668, 0.2668, 0.2667,  0.2667]
l_it_e = np.linspace(1,len(ExactWBD), len(ExactWBD)) *10
WD = res[3]
WD = [w for w in WD if w>0 ]
l_it = np.linspace(1,len(WD), len(WD))
# WD = WD[100:]
# l_it = l_it[100:]
plt.figure()
plt.plot(10*l_it, WD, '-o', label='MAM')
plt.plot(10*l_it, [0.2667]*len(WD), 'r', label='exact barycentric distance of the exact solution')
plt.plot(10*l_it_e, ExactWBD, '--g', label='exact barycentric distance of MAM')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Approached Wasserstein distance')
plt.grid()
plt.show()

distB = res[4]
distB = [w for w in distB if w>0 ]
l_it = np.linspace(1,len(distB), len(distB))
# distB = distB[100:]
plt.figure()
plt.plot(10*l_it, distB, '--')
plt.xlabel('Iterations')
plt.ylabel('Evaluation of the error')
plt.grid()
plt.show()


savemat("MATLAB/MAM_rho4000.mat", {"mam":P})

if 1==2:
    with open(f'res_altschuler/ours_exact.pkl', 'rb') as f:
        resA = pickle.load(f)

    R = 60 * 10 - 10 + 1
    pA, xedges, yedges = np.histogram2d(resA[:, 0] * R, resA[:, 1] * R, bins=R, range=[[0, R], [0, R]], weights=resA[:,2])
    pA = pA/np.sum(pA)
    pA = np.reshape(pA, (pA.size,))


    savemat("res_altschuler/res_altschuler.mat", {"Altschuler":pA})

    with open(f'res_altschuler/ibp_eps002.pkl', 'rb') as f:
        resIBP = pickle.load(f)
    resIBP[:,:2] = np.round(resIBP[:,:2]  * 60)
    R = 60
    pIBP, xedges, yedges = np.histogram2d(resIBP[:, 0], resIBP[:, 1] , bins=R, range=[[0, R], [0, R]], weights=resIBP[:,2])
    pIBP = pIBP/np.sum(pIBP)
    plt.figure()
    plt.imshow(pIBP, cmap='hot_r') #, vmax=0.0001)
    plt.colorbar()
    plt.show()
    pIBP = np.reshape(pIBP, (pIBP.size,))

    savemat("res_altschuler/res_IBP.mat", {"IBP":pIBP})


    with open(f'res_altschuler/maaipm.pkl', 'rb') as f:
        res = pickle.load(f)
    res[:,:2] = np.round(res[:,:2]  * 60)
    R = 60
    p, xedges, yedges = np.histogram2d(res[:, 0] , res[:, 1] , bins=R, range=[[0, R], [0, R]], weights=res[:,2])
    p = p/np.sum(p)
    plt.figure()
    plt.imshow(p, cmap='hot_r')
    plt.colorbar()
    plt.show()
    pIBP = np.reshape(p, (p.size,))

    savemat("res_altschuler/maaipm.mat", {"maaipm":p})


