"""
Copyright 2017 Shane T. Barratt

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import argparse
from operator import is_
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3
import os

from imagenet.dataloader import ImageDataset
from imagenet.is_tf import get_inception_score, get_inception_score_bugged

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate inception score')
    parser.add_argument('--path', type=str, default='imagenet/data/2022_01_20_12_IS_trunc_1.0/ims', help='Path to the images')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--splits', type=int, default=1, help='Number of splits')
    parser.add_argument('--resize', action='store_true', help='Resize images to 299x299 before scoring')
    parser.add_argument('--cuda', action='store_true', help='Use GPU')
    parser.add_argument('--kind', type=str, default='x_gen', help='Kind of images to use. E.g CGN Biggan or mask')
    parser.add_argument('--tensorflow', action='store_true', help='Use tensorflow')
    parser.add_argument('--bugged', action='store_true', help='Use buggy version of tensorflow')
    args = parser.parse_args()

    if args.tensorflow:
        imgs = ImageDataset(args.path, args.kind, transform = np.array)
        n = len(imgs)
        flattened_size = 256 * 256 * 3
        if args.bugged:
            dataloader = torch.utils.data.DataLoader(imgs, batch_size=1)
            imgs = []
            for i, batch in enumerate(dataloader, 0):
                imgs.append(batch.squeeze().numpy())
            is_score = get_inception_score_bugged(imgs, splits=args.splits)
        else:
            all_samples = np.zeros([int(np.ceil(float(n)/args.batch_size)*args.batch_size), flattened_size],dtype=np.uint8)
            dataloader = torch.utils.data.DataLoader(imgs, batch_size=args.batch_size, drop_last=True)
            for i, batch in enumerate(dataloader):# inception score for num_batches of real data
                all_samples[i*args.batch_size:(i+1)*args.batch_size]= batch.numpy().reshape(args.batch_size, flattened_size).astype(np.uint8)
            is_score = get_inception_score(all_samples[:n].reshape([-1,256,256,3]).transpose([0,3,1,2]), args.splits)
    else :
        print("Loading images...")
        imgs = ImageDataset(args.path, kind = args.kind, transform=transform_img)
        
        print('Computing inception score...')
        is_score = inception_score(imgs, cuda=args.cuda, batch_size=args.batch_size, resize=args.resize, splits=args.splits)
    print('Mean: {:.3f}'.format(is_score[0]))
    print('Std: {:.3f}'.format(is_score[1]))

