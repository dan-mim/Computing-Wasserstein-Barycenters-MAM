import pickle
import numpy as np
import matplotlib.pyplot as plt

# Parameters
nb_pixel_side = 80
TIME_SPENT = 100
rho = 100


## Comparison of the influence of gamma :
data = 'weights' # 'weights' #'weights' 2

gamma, eta =1000, 6*10**-3 #, 10**-3 #10**-3 # 1000
with open(f'local_MAM_10parallel_100s_M_3_100rho_{gamma}__{eta}unbalanced_{data}.pkl', 'rb') as f:  # _centersMNIST #_dataset1
    res = pickle.load(f)

with open(f'data_base_images_unbalanced_{data}.pkl', 'rb') as f:  # _centersMNIST #_dataset1
    b = pickle.load(f)
b = [b[3], b[4], b[5]]
Pi = res[3]
l_q = []
for pi_m in Pi:
    keys = pi_m.keys()
    for m in keys:
        I = b[m] > 0
        q = np.zeros(len(b[m]))
        pi = pi_m[m]
        qI = np.sum(pi, axis=0)
        q[I] = qI
        l_q.append((q,m))
        print(np.sum(b[m]) , np.sum(q))#, np.max(q/np.sum(q) - b[m]/np.sum(b[m])))

# Set up the axes
fig, axs = plt.subplots(1,3, figsize=(5, 10))  # (9,5.5) 5, 10
i = 0
for ax in axs.ravel():
    q,m = l_q[i]
    # plot:
    ax.imshow(np.reshape(q, (nb_pixel_side, nb_pixel_side)), cmap='hot_r', vmax=10**-5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"m={m}, weight={np.round(np.sum(q) , 4)}", fontsize=10.0)
    i = i + 1

plt.show()


# Set up the axes
fig, axs = plt.subplots(1,3, figsize=(5, 10))
i = 0
for ax in axs.ravel():
    q,m = b[i], i
    # plot:
    ax.imshow(np.reshape(q, (nb_pixel_side, nb_pixel_side)), cmap='hot_r', vmax=10**-5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"m={m}, weight={np.round(np.sum(q) , 4)}", fontsize=10.0) #, weight={np.round(np.sum(q) , 4)}
    i = i + 1

plt.show()