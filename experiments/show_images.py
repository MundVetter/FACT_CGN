import matplotlib.pyplot as plt
import torch
import numpy as np

datasets = ['colored_MNIST', 'double_colored_MNIST', 'wildlife_MNIST']

for dataset_name in datasets:
    path = 'mnists/data/samples' + dataset_name + ".pth"
    # path = f'mnists/data/{dataset_name}_test.pth'
    data = torch.load(path)

    nn = 20
    fig, ax = plt.subplots(2, 10, figsize = (5,5))
    plt.subplots_adjust(wspace=0.02, hspace=0.02)

    indices = list(range(0,nn))

    k = 0
    plt.set_cmap('hot')

    for i in range(2):
        for j in range(10):
            npimg = data[0][indices[k]]
            print(data[1][indices[k]])
            npimg = npimg.numpy()
            npimg = (np.transpose(npimg, (1, 2, 0)) + 1) / 2
            ax[i][j].imshow(npimg)
            ax[i][j].axis('off')
            k = k+1

    plt.show()