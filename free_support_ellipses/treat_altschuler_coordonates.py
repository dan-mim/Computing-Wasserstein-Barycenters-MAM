import matplotlib.pyplot as plt
import numpy as np
import pickle
#
# with open(f'data/ours_exact_just_computed_ellipses.pkl', 'rb') as f:
#     res = pickle.load(f)
#
# R = 60 * 10 - 10 + 1
# hist, xedges, yedges = np.histogram2d(res[:, 0] * R, res[:, 1] * R, bins=R, range=[[0, R], [0, R]])
# plt.figure()
# plt.imshow(hist, cmap='hot_r', vmax=10**-4)
# plt.show()

with open(f'data/input_ellipse_data.pkl', 'rb') as f:
    b = pickle.load(f)

b_pix = []
for i,bi in enumerate(b):
    r = 60
    R= r*r
    hist, xedges, yedges = np.histogram2d(bi[:, 0] * r, bi[:, 1] * r, bins=r, range=[[0, r], [0, r]])
    b_pix.append(np.reshape(hist,(R,)))
    # if i in [3, 5, 7]:
    #     plt.figure()
    #     plt.imshow(hist, cmap='hot_r', vmax=10**-4)
    #     plt.show()

with open(r'C:\Users\mimounid\WORK\MAM\article_Wasserstein\article\review\codes\Altschuler/'+f'dataset_altschuler.pkl','wb') as f:
    pickle.dump(b_pix, f)