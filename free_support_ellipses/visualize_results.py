import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import numpy as np


with open(f'outputs_altschuler/altschuler_exact_ellipses.pkl', 'rb') as f:
    res_altschuler = pickle.load(f)

R = 60 * 10 - 10 + 1
hist, xedges, yedges = np.histogram2d(res_altschuler[:, 0] * R, res_altschuler[:, 1] * R, bins=R, range=[[0, R], [0, R]], weights=res_altschuler[:,2])
plt.figure()
plt.imshow(hist, cmap='hot_r', vmax=0.0001)
plt.colorbar()
plt.title("Altshuler's results")
plt.show()

# Reconstruction des données
x_mesh, y_mesh = np.meshgrid(xedges[:-1], yedges[:-1], indexing='ij')
x_flat = x_mesh.flatten()
y_flat = y_mesh.flatten()
intensity_flat = hist.flatten()
I = intensity_flat>0

# Créer le tableau res reconstruit
res_reconstructed = np.column_stack((x_flat[I]/R, y_flat[I]/R, intensity_flat[I]))

try:
    # Resultats pour MAM
    name = f'outputs_MAM/saving_MAM_evry10i_3'  #M10_digit3_2000s
    with open(f'{name}.pkl', 'rb') as f:
        RES_MAM = pickle.load(f)
    continue_work = True
except:
    continue_work = False
    print("Results are not stored in outputs_MAM, run execute_MAM.py first")
    print("Presaved results after 4000 MAM iterations are displayed here: ")
    img = mpimg.imread("figures/MAM.png")
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    img = mpimg.imread("figures/evolution_MAM.png")
    plt.imshow(img)
    plt.axis('off')
    plt.title("Evolution MAM")
    plt.show()

if continue_work:
    # Parameters
    nb_pixel_side = 40
    digit = 3
    dataset = 'dataset1'  # ellipses
    l_tps0 = np.array([0.04, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    l_tps = l_tps0 *3600 # np.array([90*i for i in range(10)])*60
    # list of times
    time10 = RES_MAM[2]
    Time_MAM = []
    for i in range(len(time10)):
        Time_MAM.extend([time10[i]])
        Time_MAM.extend([10]*9)
    CumsumTime_MAM = np.cumsum(Time_MAM)
    l_i_MAM = [min(enumerate(CumsumTime_MAM), key=lambda x: abs(x[1]-tps))[0]//10 for tps in l_tps]

    # Set up the axes
    vmax = 0.00006
    fig2, axs2 = plt.subplots(2,5,figsize=(13,5)) #(9,5.5)
    # Iterate over the plotting locations
    nb_pixel_side= int(RES_MAM[0].shape[0]**.5)
    i = 0
    ll_it = []
    for ax in axs2.ravel():
        interval = l_i_MAM[i]
        it = RES_MAM[6][interval]
        ax.imshow(np.reshape(RES_MAM[1][:, interval], (nb_pixel_side, nb_pixel_side)), cmap='hot_r', vmax=vmax)
        tps = np.round(CumsumTime_MAM[int(it)] / 3600, 2)
        ax.set_title(f"k = {it}, t={l_tps0[i]}h")
        ll_it.append(it)
        ax.set_xticks([])
        ax.set_yticks([])

        i = i + 1

    fig2.subplots_adjust(bottom=0.2)
    fig2.text(0.5, 0.05, "MAM's results", ha='center', fontsize=14)
    plt.show()


    distB = RES_MAM[4]
    distB = [w for w in distB if w>0 ]
    Precision = RES_MAM[5]
    Precision = [w*10**3 for w in Precision if w>0 ]
    l_it = np.linspace(1,len(distB), len(distB))


    # Results computed with gurobi
    ExactWBD = [  0.2831, 0.2749, 0.2714, 0.2694, 0.2683,  0.2677,  0.2674, 0.2672, 0.2671,  0.2670, 0.2670, 0.2669, 0.2669, 0.2669,  0.2669, 0.2669, 0.2669, 0.2668,  0.2668, 0.2668, 0.2668, 0.2668, 0.2668, 0.2668, 0.2668, 0.2668, 0.2668,  0.2668, 0.2668, 0.2668, 0.2668, 0.2668, 0.2668, 0.2668, 0.2668, 0.2668, 0.2668,    0.2668,    0.2668,    0.2668, 0.2668, 0.2668, 0.2668, 0.2668, 0.2667,  0.2667]
    l_it_e = np.linspace(1,len(ExactWBD), len(ExactWBD)) *10
    WD = RES_MAM[3]
    WD = [w for w in WD if w>0 ]


    # Plot last figure
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('k')
    ax1.set_ylabel('Barycentric distance')
    ax1.plot(10*l_it_e, ExactWBD, '--o',  markersize=3, color='g', label=r'$\bar W_2^2(p^k)$')
    ax1.plot(10*l_it, [0.2667]*len(WD), 'r', label=r'$\bar W_2^2(p_{exact})$')
    ax1.plot(10*l_it, WD, '-o', markersize=3, label=r'$\hat W_2^2(p^k)$')
    ax1.plot(10*l_it, [0.2667]*len(WD), ':',  color='tab:blue', label=r'$\|{p^{k+1}-p^k}\|_\infty \cdot 10^3$')
    ax1.plot(10*l_it, [0.2667]*len(WD), '--', color='tab:blue', label=r'$dist_B(\hat\pi^k)$')
    ax1.plot(10*l_it, [0.2667]*len(WD), 'r') #, label=r'$\barW_2^2(p_{exact})$')
    for ip,pos in enumerate(ll_it):
        ax1.axvline(x=pos, color='#789F8C', linestyle='-.')
        ax1.text(pos, plt.ylim()[1], f'{ip}h', color='#789F8C', ha='center', va='bottom',fontsize=20)
    ax1.legend(fontsize=15)
    # ax1.rc('axes', labelsize=20)
    ax1.grid()

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel(r'Stopping criterion', color=color)
    ax2.plot(10*l_it, distB, '--', label=r'Stopping criterion')
    plt.plot(10*l_it, Precision, ':', color='tab:blue', label=r'Precision')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
