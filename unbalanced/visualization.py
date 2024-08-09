import pickle
import numpy as np
import matplotlib.pyplot as plt

# Parameters
nb_pixel_side = 80
TIME_SPENT = 100
rho = 100

# ## Evolution of the (Unbalanced) Barycenter with the iterations :
# with open(f'local_MAM_2parallel_100s_M_50_100rho_1000-0unbalanced_2.pkl', 'rb') as f:  #_centersMNIST #_dataset1
#     RES_MAM = pickle.load(f)
# mult = RES_MAM[-1]
# print('iteration nb ' , mult)
#
# # Set up the axes
# fig, axs = plt.subplots(5,5,figsize=(11,11)) #(9,5.5)
# # Iterate over the plotting locations
# i = 0
# mult = 40//25
# for ax in axs.ravel():
#     interval = i * mult
#     ax.imshow(np.reshape(RES_MAM[1][:,interval], (nb_pixel_side,nb_pixel_side)) , cmap='Greys')
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_title(f"i = {interval}")
#     i = i + 1
#
# plt.show()


## Comparison of the influence of gamma :
l_gamma = [1000] #, 300, 70, 10**-3] # [1000, 300] #, 70, 10**-3]
l_eta = [10**-4, 10**-3, 2*10**-3, 4*10**-3, 5*10**-3 , 6*10**-3 , 1, 10, 1000]
l_eta = l_eta[::-1]

# Set up the axes
fig, axs = plt.subplots(len(l_gamma), len(l_eta),figsize=(45,8)) #(9,5.5)
# Iterate over the plotting locations
i = 0
for ax in axs.ravel():
    eta = l_eta[i%len(l_eta)]
    gamma = l_gamma[i//len(l_eta)]
    with open(f'local_MAM_10parallel_100s_M_3_100rho_{gamma}__{eta}unbalanced_weights.pkl', 'rb') as f:  # __{eta} # _centersMNIST #_dataset1
        RES_MAM = pickle.load(f)
    print(f"gamma = {gamma}, eta= {eta}, iterations = {RES_MAM[-1]}, weight={np.sum(RES_MAM[0])}")
    im = ax.imshow(np.reshape(RES_MAM[0], (nb_pixel_side,nb_pixel_side)) , cmap='hot_r', vmax=0.3) #10**-3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title( r"$\gamma =$" + f" {gamma}" + r", $\eta =$" + f" {eta}" , fontsize = 10.0) #r"$\eta =$" + f" {eta}," +
    i = i + 1

# colorbar:
fig.subplots_adjust(right=0.83)
cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
fig.colorbar(im, cax=cbar_ax)

plt.show()