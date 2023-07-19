import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


class WineDataset(Dataset):

    def __init__(self, transform=None):
        # data loading
        xy = np.loadtxt('/Users/YM00VU/Desktop/Projects/test/data/wine.csv', delimiter=',', dtype=np.float32,
                        skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples


class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        #return torch.from_numpy(inputs), torch.from_numpy(targets)
        return inputs, targets


class MulTransform:
    def __init__(self, factor: int = 2):
        self.factor = factor

    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target

dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))

composed = torchvision.transforms.Compose(ToTensor())
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))