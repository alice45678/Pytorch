import torch
import torch.nn as nn
import numpy as np

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out


class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        return out

loss = nn.CrossEntropyLoss()
Y = torch.tensor([0])
# n_samples x n_classes = 1x3
sample_good = torch.tensor([[2.0, 1.0, 0.1]])
sample_bad = torch.tensor([[0.5, 2.0, 0.3]])

l1 = loss(sample_good, Y)
l2 = loss(sample_bad, Y)
print(l1.item())
print(l2.item())

_, predictions1 = torch.max(sample_good, 1)
_, predictions2 = torch.max(sample_bad, 1)
print(predictions1)
print(predictions2)

# 3 classes
loss = nn.CrossEntropyLoss()
Y = torch.tensor([2, 0, 1])
# n_samples x n_classes = 1x3
sample_good = torch.tensor([[0.1, 1.0, 3.0], [2.0, 1.0, 0.1], [0.5, 2.0, 0.1]])
sample_bad = torch.tensor([[0.5, 2.0, 0.3], [1.0, 3.0, 0.1], [2.0, 1.0, 0.1]])

l1 = loss(sample_good, Y)
l2 = loss(sample_bad, Y)
print(l1.item())
print(l2.item())

_, predictions1 = torch.max(sample_good, 1)
_, predictions2 = torch.max(sample_bad, 1)
print(predictions1)
print(predictions2)