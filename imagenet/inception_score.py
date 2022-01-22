"""
Copyright 2017 Shane T. Barratt

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import argparse
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3
from torch.utils.data import Dataset
import os
from PIL import Image

import numpy as np
from scipy.stats import entropy

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

def transform_img(img):
    # Transforms a Pil img into a numpy array scaled to [-1, 1] and moves the color channel to the last dimension
    return np.moveaxis(np.array(img), -1, 0) / 127.5 - 1

class ImageDataset(Dataset):
    def __init__(self, path, transform=None):
        """ Dataset for images in a directory

        Args:
            path (string): path to folder containing images.
            transform (callable, optional): Optional transform to be applied
        """
        self.path = path
        self.img_paths = os.listdir(path)
        self.transform = transform

    def __getitem__(self, index):
        img_path = os.path.join(self.path, self.img_paths[index])
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_paths)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate inception score')
    parser.add_argument('--path', type=str, default='imagenet/data/2022_01_20_12_IS_trunc_1.0/ims', help='Path to the images')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--splits', type=int, default=1, help='Number of splits')
    parser.add_argument('--resize', action='store_true', help='Resize images to 299x299 before scoring')
    parser.add_argument('--cuda', action='store_true', help='Use GPU')
    args = parser.parse_args()

    print("Loading images...")
    imgs = ImageDataset(args.path, transform=transform_img)
    
    print('Computing inception score...')
    is_score = inception_score(imgs, cuda=args.cuda, batch_size=args.batch_size, resize=args.resize, splits=args.splits)
    print('Mean: {:.3f}'.format(is_score[0]))
    print('Std: {:.3f}'.format(is_score[1]))

