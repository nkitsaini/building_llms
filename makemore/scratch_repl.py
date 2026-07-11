# %%
#
import torch

# %%
#
x = torch.tensor([[1, 1], [1, 2], [1, 2.]], requires_grad=True)
a = torch.tensor([[1, 2, 3], [2, 1, 2]]).float()
b = torch.tensor([[-1], [1]]).float()
c = torch.tensor([[1, 1]]).float()

c @ a @ x @ b
