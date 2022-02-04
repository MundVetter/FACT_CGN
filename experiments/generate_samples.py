from functools import lru_cache
import train_classifier
from generate_data import *
from dataloader import *
import json

import argparse
import warnings
from tqdm import trange
import torch
import repackage
repackage.up()

from mnists.train_cgn import CGN
from mnists.dataloader import get_dataloaders
from utils import load_cfg

def generate_samples(cgn, dataset_name, device):
    x, y = [], []
    cgn.batch_size = 100
    n_classes = 10

    for i in range(10):
        y_gen = torch.Tensor([i]).long().to(device)
        mask, _, _ = cgn(y_gen)

        _, foreground, background = cgn(y_gen, counterfactual=True)
        x_gen = mask * foreground + (1 - mask) * background

        x.append(x_gen.detach().cpu())
        y.append(y_gen.detach().cpu())

    for i in range(10):
        y_gen = torch.Tensor([i]).long().to(device)
        mask, _, _ = cgn(y_gen)

        _, foreground, background = cgn(y_gen, counterfactual=False)
        x_gen = mask * foreground + (1 - mask) * background

        x.append(x_gen.detach().cpu())
        y.append(y_gen.detach().cpu())

    dataset = [torch.cat(x), torch.cat(y)]
    print(f"x shape {dataset[0].shape}, y shape {dataset[1].shape}")
    torch.save(dataset, 'mnists/data/samples' + dataset_name + ".pth")

datasets = ['colored_MNIST', 'double_colored_MNIST', 'wildlife_MNIST']

for dataset in datasets:
    weights_path = "mnists/experiments/cgn_" + dataset + "/weights/ckp.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cgn = CGN()
    cgn.load_state_dict(torch.load(weights_path, 'cpu'))
    cgn.to(device).eval()

    # generate
    print(dataset)
    generate_samples(cgn=cgn, dataset_name=dataset, device=device)
