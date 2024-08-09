import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

data = pd.read_csv('emnist-letters-train.csv')

data_A = data.loc[data['23'] == 1]
liste_uppercase = [1,4,6,10,11,12,16,32,35,36,37,45,48,55,56,57,58,73,75,84,86,100,102,106,107,112,114,116,130,132,133,136,140,141,143,144,145,146,148,160,162,165,170,173,174,179,184,188,191,200]
data_A_uppercase = [list(data_A.iloc[i])[1:] for i in liste_uppercase]
data_A_uppercase = [np.reshape( np.reshape(data_A_uppercase[i], (28,28)).T, (28*28,)) for i in range(len(data_A_uppercase))]

data_M = data.loc[data['23'] == 13]
liste_uppercase = [3,7,12,21,24,32,46,55,58,61,63,68,75,76,80,87,90,95,108,114,132,136,138,140,152,159,162,175,185,188,197,198,203,209,219,224,232,234,244,248,258,303,310,340,374,385,387,406,438,533]
data_M_uppercase = [list(data_M.iloc[i])[1:] for i in liste_uppercase]
data_M_uppercase = [np.reshape( np.reshape(data_M_uppercase[i], (28,28)).T, (28*28,)) for i in range(len(data_M_uppercase))]

data_m = data.loc[data['23'] == 13]
liste_lowercase = [1,2,5,6,8,15,17,18,19,22,28,31,36,37,39,40,47,48,60,65,70,74,81,89,102,104,107,110,118,124,134,145,146,164,172,182,187,190,205,211,217,222,243,269,290,326,345,346,371,378]
data_m_lowercase = [list(data_m.iloc[i])[1:] for i in liste_lowercase]
data_m_lowercase = [np.reshape( np.reshape(data_m_lowercase[i], (28,28)).T, (28*28,)) for i in range(len(data_m_lowercase))]


# with open(f'A.pkl', 'wb') as f:
#     pickle.dump(data_A_uppercase, f)
with open(f'M.pkl', 'wb') as f:
    pickle.dump(data_M_uppercase, f)
with open(f'm_lower.pkl', 'wb') as f:
    pickle.dump(data_m_lowercase, f)

fig, axs = plt.subplots(5,9,figsize=(10,10)) #(9,5.5)
# Iterate over the plotting locations
i = 1
for ax in axs.ravel():
    ax.imshow(np.reshape(data_m_lowercase[i], (28,28)).T , cmap='Greys')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(i)
    i = i + 1

plt.show()