import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import os

"""
This function is used to visualize the MNIST data set 
generated by the CGN.
"""

# Generates the images in the given path using
def generate_images(path, data, final_labels):
    nrows, ncols = 10, 6
    fig, ax = plt.subplots(nrows, ncols, figsize = (2, 4))
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.set_cmap('hot')

    for i in range(nrows):
        indices = final_labels[i]
        for j in range(ncols):
            npimg = data[0][indices[j]]
            npimg = npimg.numpy()
            npimg = (np.transpose(npimg, (1, 2, 0)) +1) /2
            ax[i][j].imshow(npimg)
            ax[i][j].axis('off')

    return fig, ax

# Returns the indices of the labels (0,1,2,3,...) in order.
# For each digit, n_img labels are found.
def get_labels(labels, n_img):
    label0 = list(np.where(labels == 0)[0])
    label1 = list(np.where(labels == 1)[0])
    label2 = list(np.where(labels == 2)[0])
    label3 = list(np.where(labels == 3)[0])
    label4 = list(np.where(labels == 4)[0])
    label5 = list(np.where(labels == 5)[0])
    label6 = list(np.where(labels == 6)[0])
    label7 = list(np.where(labels == 7)[0])
    label8 = list(np.where(labels == 8)[0])
    label9 = list(np.where(labels == 9)[0])

    # This part is the indices of the specific labels
    labell0 = random.sample(label0, n_img)
    label1 = random.sample(label1, n_img)
    label2 = random.sample(label2, n_img)
    label3 = random.sample(label3, n_img)
    label4 = random.sample(label4, n_img)
    label5 = random.sample(label5, n_img)
    label6 = random.sample(label6, n_img)
    label7 = random.sample(label7, n_img)
    label8 = random.sample(label8, n_img)
    label9 = random.sample(label9, n_img)

    final_labels = [labell0, label1, label2, label3, label4, label5, label6, label7, label8, label9]

    return final_labels

def final_step(path, n_img):
    #Images
    data = torch.load(path)
    # Corresponding labels
    labels= data[1][:].numpy()
    # Indices of the labels
    final_labels = get_labels(labels, n_img)

    return data, final_labels


if __name__ == "__main__":

    n_img = 6
    img_path = "mnists/data/generated_images"

    # Paths for colored, double-colored and wildlife MNISTs
    path_color_ctf = "mnists/data/colored_MNIST_counterfactual.pth"
    path_color_NOT_ctf = "mnists/data/colored_MNIST_NOT_counterfactual.pth"
    path_double_color_ctf = "mnists/data/double_colored_MNIST_counterfactual.pth"
    path_double_color_NOT_ctf = "mnists/data/double_colored_MNIST_NOT_counterfactual.pth"
    path_wildlife_ctf = "mnists/data/wildlife_MNIST_counterfactual.pth"
    path_wildlife_NOT_ctf = "mnists/data/wildlife_MNIST_NOT_counterfactual.pth"

    #For the images generated using CGN we trained
    path_color_ctf_trained = "mnists/data/colored_MNIST_trained_counterfactual.pth"
    path_double_color_ctf_trained = "mnists/data/double_colored_MNIST_trained_counterfactual.pth"
    path_wildlife_ctf_trained = "mnists/data/wildlife_MNIST_trained_counterfactual.pth"

    # Visualize the images
    fig_c_ctf, ax0 = generate_images(path_color_ctf, final_step(path_color_ctf, n_img)[0], final_step(path_color_ctf, n_img)[1])
    fig_c_NOT_ctf, ax5 = generate_images(path_color_NOT_ctf, final_step(path_color_NOT_ctf, n_img)[0],final_step(path_color_NOT_ctf, n_img)[1])
    fig_dc_ctf, ax1 = generate_images(path_double_color_ctf,  final_step(path_double_color_ctf, n_img)[0],final_step(path_double_color_ctf, n_img)[1])
    fig_dc_NOT_ctf, ax2 = generate_images(path_double_color_NOT_ctf, final_step(path_double_color_NOT_ctf, n_img)[0], final_step(path_double_color_NOT_ctf, n_img)[1])
    fig_wildlife_ctf, ax3 = generate_images(path_wildlife_ctf, final_step(path_wildlife_ctf, n_img)[0],final_step(path_wildlife_ctf, n_img)[1])
    fig_wildlife_NOT_ctf, ax4 = generate_images(path_wildlife_NOT_ctf, final_step(path_wildlife_NOT_ctf, n_img)[0], final_step(path_wildlife_NOT_ctf, n_img)[1])

    #Saves the figures
    fig_c_ctf.savefig(os.path.join(img_path, 'colored_counterfactual.png'))
    fig_c_NOT_ctf.savefig(os.path.join(img_path, 'colored_NOT_counterfactual.png'))
    fig_dc_ctf.savefig(os.path.join(img_path, 'double_colored_counterfactual.png'))
    fig_dc_NOT_ctf.savefig(os.path.join(img_path, 'double_colored_NOT_counterfactual.png'))
    fig_wildlife_ctf.savefig(os.path.join(img_path, 'wildlife_counterfactual.png'))
    fig_wildlife_NOT_ctf.savefig(os.path.join(img_path, 'wildlife_NOT_counterfactual.png'))

    # Starting from this line below, MNIST images are generated using the CGN we trained.
    fig_c_ctf_trained, ax0 = generate_images(path_color_ctf_trained, final_step(path_color_ctf_trained, n_img)[0], final_step(path_color_ctf_trained, n_img)[1])
    fig_dc_ctf_trained, ax1 = generate_images(path_double_color_ctf_trained,  final_step(path_double_color_ctf_trained, n_img)[0],final_step(path_double_color_ctf_trained, n_img)[1])
    fig_wildlife_ctf_trained, ax3 = generate_images(path_wildlife_ctf_trained, final_step(path_wildlife_ctf_trained, n_img)[0],final_step(path_wildlife_ctf_trained, n_img)[1])

    fig_c_ctf_trained.savefig(os.path.join(img_path, 'colored_counterfactual_trained.png'))
    fig_dc_ctf_trained.savefig(os.path.join(img_path, 'double_colored_counterfactual_trained.png'))
    fig_wildlife_ctf_trained.savefig(os.path.join(img_path, 'wildlife_counterfactual_trained.png'))

    plt.show()
