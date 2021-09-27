import torch

batch = 10
features = 25
x = torch.rand((batch, features))

print(x[0].shape)
print(x[:, 0].shape)
print(x[2, 0:10])
x[0,0] = 100

# Fancy indexing
x = torch.rand((3,5))
rows = torch.tensor([1,0])
cols = torch.tensor([4,0])
print(x[rows, cols].shape)

# Advanced indexing
x = torch.arange(10)
print(x[(x < 2) & (x > 8)])
print(x[x.remainder(2) == 0])

# Useful operations
print(torch.where(x>5, x, x*2))
print(torch.tensor([0,0,1,1,1,2,2,3]).unique())
print(x.ndimension())
print(x.numel())