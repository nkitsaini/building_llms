# %%
#
import torch

# %%
torch.manual_seed(1337)
B, T, C =4,8,2

x = torch.randn(B, T, C)
x.shape

x

# %%

avg = torch.zeros_like(x)
for i in range(T):
    avg[:, i, :] = x[:, :i+1, :].mean(1)
# for b in range(B):
#     for i in range(T):
#         avg[b, i] = x[b, :i+1].mean(0, keepdim=True)
avg.shape

avg

# %%
M = torch.zeros(T, T)
# for i in range(T):
#     M[i, :i+1] = 1/(i+1)
# M = torch.tril(torch.ones(T, T)) * torch.tensor([1/x for x in range(1, T+1)]).view(-1, 1)
M = torch.tril(torch.ones(T, T))
M = M / M.sum(1, keepdim=True)
M
# %%

# avg
torch.allclose(M@x, avg)
diff = torch.abs(M @ x - avg).max()
diff
# M @ x
# x
# avg
