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
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model prediction
def forward(x):
    return w * x

learning_rate = 0.01
n_iters = 30
loss = nn.MSELoss()
optimizer = torch.optim.SGD([w], lr=learning_rate)
print(optimizer.state_dict())
for epoch in range(n_iters):
    # prediction
    y_pred = forward(X)
    # loss
    l = loss(Y, y_pred)
    # gradient = backward pass
    l.backward()
    # update weights
    optimizer.step()

    if epoch % 3 == 0:
       print(f'epoch{epoch + 1}: w = {w:.3f}, loss = {l:.6f}')
       print(optimizer.state)
    # zero gradients
    optimizer.zero_grad()
        # print(f'epoch{epoch + 1}: w = {w:.3f}, dw = {dw:.5f}, loss = {l:.6f}, prediction = {y_pred:.3f}')
print(f'prediction after training : f(5) = {forward(5):.3f}')
