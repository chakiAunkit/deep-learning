"""
        GENERAL PIPELINE
    1. Design model(inout, output_size, forward pass)
    2. Construct loss and optimizer
    3. Training Loop
        - forward pass - compute predictions
        - backward pass - gradients
        - update weights
"""

import torch
import torch.nn as nn

# Our function takes input and multiples 2 with it
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape

input_size = n_features
output_size = n_features

# model prediction
# model = nn.Linear(input_size, output_size)


class LinearRegression(nn.Module):

    def __init__(self, input_dims, output_dims):
        super(LinearRegression, self).__init__()
        # define layers
        self.lin = nn.Linear(input_dims, output_dims)

    def forward(self, x):
        return self.lin(x)



model = LinearRegression(input_size, output_size)
print(f'Prediction before traning: f(5) = {model(X_test).item():.3f}')

# Training
learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = model(X)

    # loss
    l = loss(Y, y_pred)

    # gradients = backwards
    l.backward()  # dl/dw

    # update weights
    optimizer.step()

    # zero grads
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')


print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')