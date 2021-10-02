import torch

x = torch.randn(3, requires_grad=True)
print(x)


"""
y = x+2
print(y)
z = y*y*2
z = z.mean()
print(z)

z.backward()
print(x.grad)

"""

# x.requires_grad_(False)
# x.detach()
# with torch.no_grad()

x.requires_grad_(False)
print(x)

# DUMMY DATA

weights = torch.ones(4, requires_grad=True)

for model in range(1): # change with 2,3,4..
    model_output = (weights*3).sum()

    model_output.backward()
    print(weights.grad)

    # weights.grad.zero() Check with and without this
