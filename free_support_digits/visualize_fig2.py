import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# Parameters
nb_pixel_side = 40
digit = 3
dataset = 'dataset1'   #centersMNIST #dataset1
l_tps = [10, 50, 500, 1000, 2000 ]

name = f'M10_digit3_b_2000s'  #M10_digit3_2000s
with open(f'{name}.pkl', 'rb') as f:
    RES_MAM = pickle.load(f)
# list of times
Time_MAM = RES_MAM[2]
CumsumTime_MAM = np.cumsum(Time_MAM)
l_i_MAM = [min(enumerate(CumsumTime_MAM), key=lambda x: abs(x[1]-tps))[0] for tps in l_tps]

# Set up the axes
vmax = 0.0002
fig2, axs2 = plt.subplots(1,5,figsize=(13,5)) #(9,5.5)
# Iterate over the plotting locations
nb_pixel_side= int(RES_MAM[0].shape[0]**.5)
i = 0
for ax in axs2.ravel():
    interval = l_i_MAM[i]
    ax.imshow(np.reshape(RES_MAM[1][:, interval], (nb_pixel_side, nb_pixel_side)), cmap='hot_r', vmax=vmax)
    ax.set_title(f"i = {interval}, t={np.round(CumsumTime_MAM[interval])}s")
    ax.set_xticks([])
    ax.set_yticks([])

    i = i + 1
# plt.title('MAM')
plt.show()

# List of probabilities
with open('digit_datasets/b_1_mat.pkl', 'rb') as f:  # b_centers_MNIST #b_1_mat
    l_b = pickle.load(f)
N = 10
b = l_b[digit]
b = b[:N]
# Set up the axes
vmax = 0.0002
fig2, axs2 = plt.subplots(2,5,figsize=(13,5)) #(9,5.5)
# Iterate over the plotting locations
nb_pixel_side= int(b[0].shape[0]**.5)
i = 0
for ax in axs2.ravel():
    ax.imshow(np.reshape(b[i], (nb_pixel_side, nb_pixel_side)), cmap='hot_r') #, vmax=vmax)
    # ax.set_title(f"i = {interval}, t={np.round(CumsumTime_MAM[interval])}s")
    ax.set_xticks([])
    ax.set_yticks([])

    i = i + 1
# plt.title('MAM')
plt.show()