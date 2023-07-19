# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#   - forward pass: compute prediction
#   - backward pass: gradients
#   - update weights

import torch
import numpy as np
import torch.nn as nn

# f = w * x
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
X_test = torch.tensor([5], dtype=torch.float32)
n_samples, n_features = X.shape
print(type(X))
print(n_samples, n_features)
input_size = n_features
output_size = n_features
# w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model prediction
#def forward(x):
#    return w * x
# model = nn.Linear(input_size, output_size)

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)
learning_rate = 0.01
n_iters = 30
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


for epoch in range(n_iters):
    # prediction
    y_pred = model(X)
    # loss
    l = loss(Y, y_pred)
    # gradient = backward pass
    l.backward()
    # update weights
    optimizer.step()

    if epoch % 3 == 0:
        [w, b] = model.parameters()
        print(f'epoch{epoch + 1}: w = {w[0][0].item():.3f}, loss = {l:.6f}')

    # zero gradients
    optimizer.zero_grad()
        # print(f'epoch{epoch + 1}: w = {w:.3f}, dw = {dw:.5f}, loss = {l:.6f}, prediction = {y_pred:.3f}')
print(f'prediction after training : f(5) = {model(X_test)}')
