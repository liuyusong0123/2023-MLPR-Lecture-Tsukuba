import torch
x = torch.tensor(2.0, requires_grad=True)
y = torch.sigmoid(x)
y.backward()
print(x.grad)
