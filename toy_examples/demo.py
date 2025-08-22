# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 12:43:38 2023

@author: mimounid
"""

## Imports:
# Basics
import pickle
# My code MAM:
from mam import MAM


## Test 1 with centered digits
N = 10
with open('toy_examples/digit_datasets/3_centered.pkl', 'rb') as f:  # b_centers_MNIST #b_1_mat
    l_b = pickle.load(f)
b = l_b[:N]

MAM(b, rho=.1,
    exact=False,
    computation_time=20, iterations_max=400, precision=10 ** -6,
    visualize=True, name=f'outputs_Centered.pkl')



## Test 2 with randomly placed and scaled digits
N = 10
with open('toy_examples/digit_datasets/3_randomPosScaled.pkl', 'rb') as f:  # b_centers_MNIST #b_1_mat
    l_b = pickle.load(f)
b = l_b[:N]

MAM(b, rho=.1,
    exact=False,
    computation_time=20, iterations_max=400, precision=10 ** -6,
    visualize=True, name=f'outputs_Centered.pkl')

