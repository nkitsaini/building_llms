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

# %% Create training data
block_size = 3
X, Y = create_training_data(words, block_size, debug = False)
X.shape, X.dtype, Y.shape, Y.dtype
# %%
embed_size = 2
device = torch.device(0)
g = torch.Generator(device).manual_seed(79783)
C = torch.randn(27, embed_size, generator = g, requires_grad=True, device= device)
W1 = torch.randn(block_size * embed_size, 100, generator = g, requires_grad=True, device= device)
B1 = torch.randn(100, generator = g, requires_grad=True, device= device)
W2 = torch.randn(100, 27, generator = g, requires_grad=True, device= device)
B2 = torch.randn(27, generator = g, requires_grad=True, device= device)
parameters = (C, W1, B1, W2, B2)
sum(n.nelement() for n in parameters)

# %%
#
X = X.to(device)
Y = Y.to(device)
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

for i in range(1000):
    loss = calc_loss(X, Y)
    for p in parameters:
        p.grad = None
    if i % 50 == 0:
        print(f"[{i}]", loss)
    loss.backward()
    for p in parameters:
        p.data -= p.grad * 0.1 # type: ignore


# xenc[0]
# xenc[0] @ C

# C[0]
# %%
#

def predict() -> str:
    word = START_TOKEN*block_size
    while True:
        x = torch.tensor([char_to_int(c) for c in word[-3:]], device=device).view(1, 3) # (1, 3)
        logits = calc_logits(x) # (1, 27)
        l = logits.exp()
        prob = l/l.sum(1, keepdim=True)
        y = torch.multinomial(prob, 1).item() # (1, 1)
        y_char =  CHARS[y]
        if y_char == END_TOKEN:
            return word[3:]
        word += y_char

for i in range(20):
    print(predict())




# %%
X.shape
# %%
#
C[X].flatten(1, 2).shape
# C[xs].unbind(1)

# %%
#
emb = C[X].view(-1, 6)
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

mean_prob = (Y.float() @ prob).log().mean()
loss = - mean_prob

loss
