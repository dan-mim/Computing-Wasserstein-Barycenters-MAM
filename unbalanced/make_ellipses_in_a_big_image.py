
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.ndimage import zoom

def lasso(b, gamma):
    """
    Lasso regularization to make data sparser
    """
    b_lasso = np.zeros(b.shape)
    for i in range(len(b)):
        elem = b[i]
        if elem > gamma:
            b_lasso[i] = elem - gamma
        elif elem < - gamma:
            b_lasso[i] = elem + gamma
    return(b_lasso)


def calculateEllipse(x, y, a, b, angle, steps):
    """
    This function calculates ellipses
    """
    beta = - angle * (np.pi / 180)
    sinbeta = np.sin(beta)
    cosbeta = np.cos(beta)

    alpha = np.linspace(0, 360, steps).T * (np.pi / 180)
    sinalpha = np.sin(alpha)
    cosalpha = np.cos(alpha)

    X = x + (a * cosalpha * cosbeta - b * sinalpha * sinbeta)
    Y = y + (a * cosalpha * sinbeta + b * sinalpha * cosbeta)

    return (X, Y)


def trace_ellispses(M, nb_ellipses, factor=10, unbalanced=False):
    """
    This function trace M pictures of nb_ellipses per images.
    M : (int) number of pictures generated
    nb_ellipses: (int) number of ellipses on 1 picture
    factor: (float) tuning parameter thickness of the ellipses
    """
    nb_ellipses = nb_ellipses + 1
    liste_ellipses = []
    for m in range(M):
        # keep track
        X = np.zeros((500, nb_ellipses))
        Y = np.zeros((500, nb_ellipses))

        a1 = factor * (8 * np.random.rand() + 5)
        b1 = factor * (8 * np.random.rand() + 5)
        angle = 360 * np.random.rand()
        p10, p11 = calculateEllipse(0, 0, a1, b1, angle, 500)
        X[:, 0] = p10
        Y[:, 0] = p11

        for Sk in range(1, nb_ellipses):
            a2 = factor * (8 * np.random.rand() + 5)
            b2 = factor * (8 * np.random.rand() + 5)
            x = np.random.rand()
            y = np.random.rand()
            scalx = (np.random.rand() - .5) * .8
            scaly = (np.random.rand() - .5) * .8
            angle = 360 * np.random.rand()
            p20, p21 = calculateEllipse(scalx * a1, scaly * b1, a2/4, b2/4, angle, 500)
            X[:, Sk] = p10
            Y[:, Sk] = p11
            a1 = a2
            b1 = b2
            p10 = p20.copy()
            p11 = p21.copy()

        xoffset = np.min(X)
        yoffset = np.min(Y)
        X = X - xoffset + 1
        Y = Y - yoffset + 1

        xmax = int(np.max(X))
        ymax = int(np.max(Y))
        IMG = np.zeros((xmax + 1, ymax + 1))
        for Sk in range(nb_ellipses):
            for i in range(500):
                IMG[int(X[i, Sk]), int(Y[i, Sk])] = 1

        # resize picture
        res = resize(IMG, (40, 40))

        if not unbalanced:
            res = res / np.sum(res)

        # File liste_ellipses:
        liste_ellipses.append(np.reshape(res, (1600,)) )

    # Output:
    return (liste_ellipses)


M = 400
b_ellipses = trace_ellispses(M, 2, factor=10, unbalanced=False)

    # with open(f'data_base_ellipses.pkl', 'wb') as f:
    #     pickle.dump(b_ellipses, f)

nb_pixel_side = 40

image_ellipses = []
for j in range(M//4) :
    i = j * 4
    # Create an 80x80 array filled with zeros
    full_image = np.zeros((80, 80))

    # Assign the resized image to the top right corner of the 80x80 array
    # Define the desired size of the resized image
    new_size = 20
    nb_of_filled_squares = np.random.choice([1, 2, 3]) #, 4
    chosen_squares = []
    for n in range(nb_of_filled_squares) :
        chosen_squares.append( np.random.choice([1, 2, 3]) ) #, 4
    if 1 in chosen_squares:
        image = np.reshape(b_ellipses[i], (nb_pixel_side, nb_pixel_side))
        resized_image = zoom(image, (new_size / 40, new_size / 40))
        full_image[10:30, 10:30] = resized_image
    if 2 in chosen_squares :
        image = np.reshape(b_ellipses[i+1], (nb_pixel_side, nb_pixel_side))
        resized_image = zoom(image, (new_size / 40, new_size / 40))
        full_image[50:70 , 10:30] = resized_image
    if 3 in chosen_squares :
        image = np.reshape(b_ellipses[i+2], (nb_pixel_side, nb_pixel_side))
        resized_image = zoom(image, (new_size / 40, new_size / 40))
        full_image[50:70, 50:70] = resized_image
    if 4 in chosen_squares :
        image = np.reshape(b_ellipses[i+3], (nb_pixel_side, nb_pixel_side))
        resized_image = zoom(image, (new_size / 40, new_size / 40))
        full_image[10:30, 50:70] = resized_image

    # new_size = 40
    # full_image = zoom(full_image, (new_size / 80, new_size / 80))


    # normalize:
    full_image = np.reshape(full_image, (80*80,))
    full_image = np.abs(full_image)
    seuil = 10**-3 # 5 * 10**-4 #
    I_1 = full_image > seuil
    full_image_01 = np.zeros(full_image.shape)
    full_image_01[I_1] = 1
    # full_image_01 = full_image_01 / np.sum(full_image_01)

    # save
    image_ellipses.append(full_image_01)

# with open(f'data_base_images_unbalanced_2.pkl', 'rb') as f:  #_centersMNIST #_dataset1
#     image_ellipses = pickle.load(f)

# Set up the axes
fig, axs = plt.subplots(10,10 ,figsize=(10,10)) #(9,5.5)
# Iterate over the plotting locations
i = 0
for ax in axs.ravel():
    ax.imshow(np.reshape(image_ellipses[i], (80,80)) , cmap='Greys')  #, vmax=vmax
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_title(f"i = {interval}, t={np.round(CumsumTime[interval])}")
    i = i + 1
# plt.title('IBP')
plt.show()

# # print(len(image_ellipses))
with open(f'data_base_images_unbalanced_weights.pkl', 'wb') as f:
        pickle.dump(image_ellipses, f)
