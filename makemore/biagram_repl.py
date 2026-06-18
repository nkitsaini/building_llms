# %%
# import os
# os._exit(00)

# %%
%load_ext autoreload
%autoreload 0

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

words = open("names.txt").read().splitlines()
len(words), words[:5]
# %%

biagrams: dict[tuple[str, str], int] = {}

for w in words:
    chars = [START_TOKEN, *w, END_TOKEN]
    for (
        c1,
        c2,
    ) in zip(chars, chars[1:]):
        biagram = (c1, c2)
        biagrams[biagram] = biagrams.get(biagram, 0) + 1


# %%
N = sorted(biagrams.items(), key=lambda i: i[1], reverse=True)[:5]
N
# %%
import string
N = create_count_tensor(words)

N[char_to_int('n')][char_to_int(END_TOKEN)]
# %%
plt.imshow(N)

# %%
plt.figure(figsize=(16, 16))
plt.imshow(N, cmap='Blues')


for i in range(len(CHARS)):
    for j in range(len(CHARS)):
        chstr = CHARS[i] + CHARS[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, str(N[i, j].item()), ha="center", va="top", color='gray')

# %%
N[0, :]

p = N[0].float()
p = p/p.sum()

plt.plot(p)
# %%

g = torch.Generator().manual_seed(2147483647)

i = torch.multinomial(p, num_samples=1, generator=g, replacement=True)
i
CHARS[i]

# %%
P = N.float()
P /= P.sum(1, keepdim=True)

# %%

# %%

g = torch.Generator().manual_seed(2147483647)


def gen_word(g: torch.Generator):
    idx = 0
    word = []
    while True:
        word.append(CHARS[idx])
        p = P[idx]
        idx = int(torch.multinomial(p, num_samples=1, generator=g, replacement=True).item())
        if idx == 0:
            break
    return word


[''.join(gen_word(g)) for _ in range(30)]
# %% Training data prep

xs, ys = create_training_data(words)

xs[:3], ys[:3]

xenc = F.one_hot(xs, 27).float()

# xenc.shape
plt.imshow(xenc[:20])

# %%
import math
def calc_matrix_loss(mat: torch.Tensor):
    total_prob = 0
    total_count = 0
    for k, count in biagrams.items():
        prob = mat[char_to_int(k[0])][char_to_int(k[1])]
        total_prob += math.log(prob)*count
        total_count += count
    return -total_prob/total_count

calc_matrix_loss(P)


# %%
#


g = torch.Generator().manual_seed(2147483647)

# class Model(nn.Module)
W = torch.randn((27, 27), requires_grad=True, generator=g)
B = torch.randn(27)

logits = (xenc[:17] @ W)  + B
counts = logits.exp()
prob = counts /counts.sum(1, keepdim=True)
prob

def calc_loss(xs: torch.Tensor, ys: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    xenc = F.one_hot(xs, 27).float()
    preds = (xenc @ weights).exp()
    probs = preds/preds.sum(1, keepdim=True)
    preds_of_next = probs[torch.arange(probs.size(0)), ys]
    loss = preds_of_next.log().mean().neg() + 0.01 * (W**2).mean()
    return loss

def update_weights(weights: torch.Tensor, lr: float = 1e-1):
    weights.data -= weights.grad * lr
loop = 200
for i in range(loop):
    W.grad = None
    loss = calc_loss(xs, ys, W)
    loss.backward()
    print(loss)
    update_weights(W, (loop-i)/3.5)
loss = calc_loss(xs, ys, W)
loss
# %%
