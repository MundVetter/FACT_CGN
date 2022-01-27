import argparse
import repackage
repackage.up()

import torch
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import numpy as np

from mnists.models.classifier import CNN
from mnists.dataloader import get_tensor_dataloaders, TENSOR_DATASETS
import pathlib

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # stats
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def do_cam(model, device, test_loader):
    model.eval()
    cam = GradCAM(model, [model.model[-4]], use_cuda=device.type == 'cuda')

    pathlib.Path(f'mnists/data/grad_cam/{args.dataset}/').mkdir(parents=True, exist_ok=True)
    for i, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        grayscale_cam = cam(data)
        
        for j, (c, img, target) in enumerate(zip(grayscale_cam, data,target)):
            path = f'mnists/data/grad_cam/{args.dataset}/{i}_{j}_{target}'
            img = np.clip(rgb2gray(img.permute(1, 2, 0).cpu().numpy().squeeze()), 0, 1)
            img = np.dstack((img, img, img))

            vis = show_cam_on_image(img, c)
            plt.imsave(f'{path}_overlay.png', vis)
            plt.imsave(f'{path}.png', c)



def main(args):
    # model and dataloader
    model = CNN()
    dl_train, dl_test = get_tensor_dataloaders(args.dataset, args.batch_size)

    # Optimizer
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # push to device and train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, dl_train, optimizer, epoch)
        test(model, device, dl_test)
        scheduler.step()
        # if epoch == 1:
    if args.grad_cam:
        do_cam(model, device, dl_test, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="double_colored_MNIST", choices=TENSOR_DATASETS,
                        help='Provide dataset name.')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--grad_cam', action='store_true', help='Use Grad-CAM')
    args = parser.parse_args()

    print(args)
    main(args)
