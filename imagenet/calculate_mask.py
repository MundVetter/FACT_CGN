import argparse
import numpy as np

from imagenet.dataloader import ImageDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='imagenet/data/2022_01_22_13_IS_big_trunc_1.0_/ims', help='Data directory containing the masks')
    args = parser.parse_args()

    print("Loading masks")
    imgs = ImageDataset(args.path, 'mask', transform= lambda x: np.array(x) / 255)

    print('Computing mean and std pixel values...')
    means = []
    var = []
    for img in imgs:
        means.append(np.mean(img))
        var.append(np.var(img))

    print('Mean: {:.3f}'.format(np.mean(means)))
    print('Std: {:.3f}'.format(np.sqrt(np.mean(var))))
