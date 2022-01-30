import repackage
repackage.up()

import torch
from mnists.dataloader import DoubleColoredMNIST

test = DoubleColoredMNIST(train=False, random=False)
dl = []
index_order = [18, 16, 11, 0, 5, 6, 2, 21, 1]
for i in index_order:
    for j in range(10):
        dl.append(test.__getitem__(i, j))
x, y = [], []
for data in dl:
    x.append(data['ims'].cpu())
    y.append(data['labels'].cpu())

torch.save([torch.stack(x), torch.as_tensor(y)], 'mnists/data/double_colored_MNIST_test.pth')