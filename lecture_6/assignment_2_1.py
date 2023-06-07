import torch
import torch.nn.functional as F
X = torch.tensor([[-1, 1], [1, 1]], dtype=torch.float32)
Y = torch.tensor([0, 1], dtype=torch.float32)
w = torch.tensor([100, 0], dtype=torch.float32, requires_grad=True)
prediction = X @ w
loss = F.binary_cross_entropy_with_logits(prediction, Y)
print(loss)
loss.backward()
print(w.grad)