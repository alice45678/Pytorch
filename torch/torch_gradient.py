import torch

# f = w * x
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model prediction
def forward(x):
    return w * x

# loss = MSE
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()

# gradient
def gradient(x, y, y_predicted):
    return (2 * x(y_predicted - y)).mean()

# training
learning_rate = 0.01
n_iters = 30
for epoch in range(n_iters):
    # prediction
    y_pred = forward(X)
    # loss
    l = loss(Y, y_pred)
    # gradient = backward pass
    l.backward()
    # update weights
    with torch.no_grad():
        w -= learning_rate * w.grad

    if epoch % 3 == 0:
       print(f'epoch{epoch + 1}: w = {w:.3f}, dw = {w.grad:.5f}, loss = {l:.6f}')
    # zero gradients
    w.grad.zero_()
        # print(f'epoch{epoch + 1}: w = {w:.3f}, dw = {dw:.5f}, loss = {l:.6f}, prediction = {y_pred:.3f}')
print(f'prediction after training : f(5) = {forward(5):.3f}')

