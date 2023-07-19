import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


class LoadData(Dataset):
    "Loads dataset"
    def __int__(self):
        # data loading
        xy = np.loadtxt('/Users/YM00VU/Desktop/Projects/test/data/wine.csv', delimiter=',', dtype=np.float32,
                        skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


dataset = LoadData()
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)

dataiter = iter(dataloader)
data = dataiter.next()
features, labels = data
print(features, labels)

# training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward, backward, update
        if (i+1) % 5 == 0:
            print(f'epoch {epoch++1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')

# build in dataset

torchvision.datasets.MNIST()
# fashion-mnist, cifar, coco
