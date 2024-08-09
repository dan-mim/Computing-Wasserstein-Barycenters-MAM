from LP_Wasserstein_distance import *
from work_distance_matrix import *

digit=3
N = 10
# List of probabilities
with open('digit_datasets/b_centers_MNIST.pkl', 'rb') as f:  # b_centers_MNIST #b_1_mat
    l_b = pickle.load(f)
b = l_b[digit]
b = b[:N]

with open(f'inexactMAM_M10_digit3_500s.pkl', 'rb') as f:
    res_inexact = pickle.load(f)
with open(f'M10_digit3_500s.pkl', 'rb') as f:
    res_exact = pickle.load(f)
p = res_exact[0]

# compute distance matrix
R = len(p)
S =len(b[0])
eps_R = 1 / R ** .5
eps_S = 1 / S ** .5

M_dist = distance_matrix(R, b[0]>-10, eps_R, eps_S)

WD = Wasserstein_distance_LP(p, b[0], M_dist)