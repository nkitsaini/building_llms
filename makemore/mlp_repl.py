# %%
%load_ext autoreload

%autoreload 2

# %% Imports - external libs

import random
from IPython.display import display, Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Sequence
from makemore.biagram import CHARS, END_TOKEN, START_TOKEN, char_to_int, create_count_tensor, create_training_data
%matplotlib inline

import numpy as np

import torch

# %%
words = open("./makemore/names.txt").read().splitlines()
len(words), words[:5]
# %%
device = torch.device(0)
# device = torch.device('cpu')

# %% Create training data
block_size = 3
import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = create_training_data(words[:n1], block_size, debug = False)
Xdev, Ydev = create_training_data(words[n1:n2], block_size, debug = False)
Xte, Yte = create_training_data(words[n2:], block_size, debug = False)
Xtr = Xtr.to(device)
Ytr = Ytr.to(device)
Xdev = Xdev.to(device)
Ydev = Ydev.to(device)
Xte = Xte.to(device)
Yte = Yte.to(device)
Xtr.shape, Xtr.dtype, Ytr.shape, Ytr.dtype
# %%


# %%
embed_size = 10

g = torch.Generator(device).manual_seed(2147483647)
C = torch.randn(27, embed_size, generator = g, requires_grad=True, device= device)
W1 = torch.randn(block_size * embed_size, 200, generator = g, requires_grad=True, device= device)
B1 = torch.randn(200, generator = g, requires_grad=True, device= device)
W2 = torch.randn(200, 27, generator = g, requires_grad=True, device= device)
B2 = torch.randn(27, generator = g, requires_grad=True, device= device)
parameters = (C, W1, B1, W2, B2)
sum(n.nelement() for n in parameters)

# %%
def calc_logits(xs: torch.Tensor) -> torch.Tensor:
    n = len(xs)
    # xs = (n, block_size)
    # C = (n_chars, embed_size)
    xenc = C[xs] # (n, block_size, embed_size)
    xenc = xenc.view(-1, block_size * embed_size) # (n, block_size * embed_size)
    h1 = torch.tanh((xenc @ W1) + B1) # (n, hidden_size)
    return (h1 @ W2) + B2 # (n, 27)

def calc_loss(xs: torch.Tensor, ys: torch.Tensor):
    return F.cross_entropy(calc_logits(xs), ys)
    # logexp = logits.exp()
    # prob = logexp/logexp.sum(1, keepdim=True)
    # loss = - prob[torch.arange(n), ys].log().mean() # (n)
    # return loss

lossi = []

# %%

batch_size = 32

total_loops = 50000
total_checkpoints = 10
lr = 0.1
lre = torch.linspace(-3, 0, total_loops)
lrs = 10**lre
lri = []

for i in range(total_loops):
    minibatch_idx = torch.randint(0, Xtr.size(0), (batch_size,))
    minibatch_x = Xtr[minibatch_idx]
    minibatch_y = Ytr[minibatch_idx]
    loss = calc_loss(minibatch_x, minibatch_y)
    for p in parameters:
        p.grad = None
    if i % (total_loops//total_checkpoints) == 0:
        print(f"[{i}]", loss)
    loss.backward()
    for p in parameters:
        p.data -= p.grad * lr # type: ignore
    # lri.append(lre[i].item())
    lossi.append(loss.log10().item())

# %%
plt.plot(range(0, total_loops), lossi)

# %%
calc_loss(Xtr, Ytr)
# %%
calc_loss(Xdev, Ydev)

# %%

plt.figure(figsize=(8,8))
# C
plt.scatter(C[:, 0].cpu().detach().numpy(), C[:, 1].cpu().detach().numpy(), s=200)
for i in range(C.size(0)):
    plt.text(C[i, 0].item(), C[i, 1].item(), CHARS[i], ha="center", va="center", color="white")
plt.grid(True,which='minor')

# %%

def predict() -> str:
    word = START_TOKEN*block_size
    while True:
        x = torch.tensor([char_to_int(c) for c in word[-block_size:]], device=device).view(1, block_size) # (1, 3)
        logits = calc_logits(x) # (1, 27)
        l = logits.exp()
        prob = l/l.sum(1, keepdim=True)
        y = torch.multinomial(prob, 1).item() # (1, 1)
        y_char =  CHARS[y]
        if y_char == END_TOKEN:
            return word[block_size:]
        word += y_char

for i in range(20):
    print(predict())
# %%
Xtr[torch.randint(0, Xtr.size(0), (5,))]
# xenc[0]
# xenc[0] @ C

# C[0]




# %%
Xtr.shape
# %%
#
C[Xtr].flatten(1, 2).shape
# C[xs].unbind(1)

# %%
#
emb = C[Xtr].view(-1, 6)
W1 =torch.randn(6, 100)
B1 =torch.randn(100)

h = torch.tanh(((emb @ W1) + B1))
h

# %%


W2 = torch.randn(100, 27)
B2 = torch.randn(27)

logits =(h @ W2) + B2

logits.shape # (32, 27)
# %%

logexp = logits.exp()
prob = logexp/logexp.sum(1, keepdim=True) # error
# %%
# ys.shape # (32)
prob.shape # (32, 27)
prob.shape # (32, 27)

mean_prob = (Ytr.float() @ prob).log().mean()
loss = - mean_prob

loss
