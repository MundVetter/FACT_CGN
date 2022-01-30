import argparse
import os
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='double_colored_MNIST', help='Dataset to use')
    parser.add_argument('--original', action='store_true', default=False, help='Infuse originals into the dataset')

    args = parser.parse_args()
    path = f'mnists/data/grad_cam/{args.dataset}_{args.original}_false/'

    if not os.path.exists(path):
        print("Grad cam folder for this dataset does not exist.")
        exit(1)
    file_paths = os.listdir(path)
    originals = [file for file in file_paths if 'img' in file]
    data_index = [0, 11, 22, 33, 44, 55, 66, 77, 88, 99]
    upper_index = [0, 1, 12, 23, 34, 45, 56, 67, 78, 89]
    lower_index = [0, 2, 14, 26, 38, 50, 62, 74, 86, 98]
    heatmaps = [path for path in file_paths if len(path.split('_')) == 3]
    heatmap_indices = [data_index] + [range(index + 1, index + 11) for index in data_index if index < 99]


    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols=(11, 11))
    img_paths_ordered = [originals[i] for i in lower_index]
    for i in range(10):
        img_paths_ordered.append(originals[upper_index[i]])
        img_paths_ordered += [heatmaps[index] for index in heatmap_indices[i]]

    img_list = [np.zeros((32, 32, 3))] + [Image.open(path + p) for p in img_paths_ordered]
    for ax, im in zip(grid, img_list):
    # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.axis('off')

    plt.show()