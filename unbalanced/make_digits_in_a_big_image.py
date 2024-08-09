
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.ndimage import zoom


M = 400
with open('b_1_mat.pkl',
          'rb') as f:  # b_centers_MNIST #b_1_mat  data_base_images_unbalanced #data_base_images_unbalanced_40_40 #data_base_images_unbalanced_2 #data_base_images_unbalanced_weights
    b = pickle.load(f)
b = b[3][:M]


    # with open(f'data_base_ellipses.pkl', 'wb') as f:
    #     pickle.dump(b_ellipses, f)

nb_pixel_side = 40

images_digits = []
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
        image = np.reshape(b[i], (nb_pixel_side, nb_pixel_side))
        resized_image = zoom(image, (new_size / 40, new_size / 40))
        full_image[10:30, 10:30] = resized_image
    if 2 in chosen_squares :
        image = np.reshape(b[i+1], (nb_pixel_side, nb_pixel_side))
        resized_image = zoom(image, (new_size / 40, new_size / 40))
        full_image[50:70 , 10:30] = resized_image
    if 3 in chosen_squares :
        image = np.reshape(b[i+2], (nb_pixel_side, nb_pixel_side))
        resized_image = zoom(image, (new_size / 40, new_size / 40))
        full_image[50:70, 50:70] = resized_image
    if 4 in chosen_squares :
        image = np.reshape(b[i+3], (nb_pixel_side, nb_pixel_side))
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
    full_image_01 = full_image_01 / np.sum(full_image_01)

    # save
    images_digits.append(full_image_01)

# with open(f'data_base_images_unbalanced_2.pkl', 'rb') as f:  #_centersMNIST #_dataset1
#     images_digits = pickle.load(f)

# Set up the axes
fig, axs = plt.subplots(10,10 ,figsize=(10,10)) #(9,5.5)
# Iterate over the plotting locations
i = 0
for ax in axs.ravel():
    ax.imshow(np.reshape(images_digits[i], (80,80)) , cmap='Greys')  #, vmax=vmax
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_title(f"i = {interval}, t={np.round(CumsumTime[interval])}")
    i = i + 1
# plt.title('IBP')
plt.show()

# # # print(len(images_digits))
# with open(f'data_digit_centered_unbalanced.pkl', 'wb') as f:
#         pickle.dump(images_digits, f)
