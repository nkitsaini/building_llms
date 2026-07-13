# %%
from IPython import get_ipython

_ip = get_ipython()
if _ip:
    _ip.run_line_magic("load_ext", "autoreload")
    _ip.run_line_magic("autoreload", "2")


# %% Imports - external libs

from tqdm import tqdm
import random
from IPython.display import display, Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Sequence
from makemore.biagram import (
    CHARS,
    END_TOKEN,
    START_TOKEN,
    char_to_int,
    create_count_tensor,
    create_training_data,
    read_words,
)
from makemore.nn import cross_entropy, Linear, Model, Tanh, BatchNorm

_ip.run_line_magic("matplotlib", "inline")

import numpy as np

import torch

# %%
words = read_words()
len(words), words[:5]
# %%
# device = torch.device(0)
device = torch.device("cpu")

# %% Create training data
block_size = 3
import random

random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = create_training_data(words[:n1], block_size, debug=False)
Xdev, Ydev = create_training_data(words[n1:n2], block_size, debug=False)
Xte, Yte = create_training_data(words[n2:], block_size, debug=False)
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

g = torch.Generator(device).manual_seed(2147483649)

hidden_size = 100
C = torch.randn(
    len(CHARS), embed_size, generator=g
)  # ; Linear(len(CHARS), embed_size, generator=g)
layers = [
    Linear(block_size * embed_size, hidden_size, bias=False),
    BatchNorm(hidden_size),
    Tanh(),
    Linear(hidden_size, hidden_size, bias=False),
    BatchNorm(hidden_size),
    Tanh(),
    Linear(hidden_size, hidden_size, bias=False),
    BatchNorm(hidden_size),
    Tanh(),
    Linear(hidden_size, hidden_size, bias=False),
    BatchNorm(hidden_size),
    Tanh(),
    Linear(hidden_size, hidden_size, bias=False),
    BatchNorm(hidden_size),
    Tanh(),
    Linear(hidden_size, len(CHARS), bias=False),
    BatchNorm(len(CHARS)),
]

with torch.no_grad():
    # assert isinstance(layers[-1], Linear)
    assert isinstance(layers[-1], BatchNorm)
    layers[-1].gamma *= 0.1
    for l in layers[:-1]:
        if isinstance(l, Linear):
            l.w *= 1

parameters = [C] + [p for l in layers for p in l.parameters()]
for p in parameters:
    p.requires_grad = True
print(list(p.nelement() for p in parameters))
print(sum(p.nelement() for p in parameters))
# %%


def forward(x: torch.Tensor, y: torch.Tensor, training: bool = True):
    xenc = C[x].view(x.size(0), -1)  # (n, 3 * 27)
    ypred = xenc
    for p in layers:
        p.training = training
        ypred = p(ypred)
    loss = cross_entropy(ypred, y)
    return loss


def backprop(loss: torch.Tensor, lr: float = 1e-1):
    for l in layers:
        l.out.retain_grad()
    for p in parameters:
        p.grad = None
    loss.backward()
    for p in parameters:
        assert p.grad is not None
        p.data -= p.grad * lr


losses = []


ud = []


def train(xs: torch.Tensor, ys: torch.Tensor, batch_size: int = 32, loops: int = 10000):
    assert len(xs) == len(ys)
    for loop_num in tqdm(range(loops)):
        batch_idx = torch.randint(len(xs), size=(batch_size,))
        x = xs[batch_idx]
        y = ys[batch_idx]
        loss = forward(x, y)
        if loop_num < 10:
            print(loss.item())
        losses.append(loss.item())
        lr = 1e-1
        backprop(loss, lr)
        with torch.no_grad():
            ud.append(
                [(lr * p.grad.std() / p.data.std()).log10().item() for p in parameters]
            )
        if loop_num > 1000:
            break


# %%
losses = []
train(Xtr, Ytr)

# %%
plt.plot(losses)
losses[:5], losses[-5:]
# %% Visualize tanh outputs

plt.figure(figsize=(20, 4))
legends = []
for i, l in enumerate(layers):
    if not isinstance(l, Tanh):
        continue
    mean = l.out.mean()
    std = l.out.std()
    saturation = 100 * (l.out.abs() > 0.97).sum() / l.out.nelement()
    print(
        f"Layer {i} ({l.__class__.__name__:>10}): mean {mean:>5.2f}, std {std:.2f}, saturation {saturation:>6.2f}%"
    )
    hy, hx = torch.histogram(l.out, density=True)
    plt.plot(hx[:-1].cpu().detach(), hy.cpu().detach())
    legends.append(f"layer {i}")


plt.legend(legends)

# %% Visualize tanh grads

plt.figure(figsize=(20, 4))
legends = []
for i, l in enumerate(layers):
    if not isinstance(l, Tanh):
        continue
    t = l.out.grad
    assert isinstance(t, torch.Tensor)
    mean = t.mean()
    std = t.std()
    print(f"Layer {i} ({l.__class__.__name__:>10}): mean {mean:>12.8f}, std {std:.8f}")
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].cpu().detach(), hy.cpu().detach())
    legends.append(f"layer {i}")
plt.legend(legends)

# %% Weight / Grad ratio

plt.figure(figsize=(20, 4))
legends = []
for i, p in enumerate(parameters):
    if p.ndim != 2:
        continue
    assert p.grad is not None
    t = p.grad  # /p.data
    mean = t.mean()
    std = t.std()
    print(
        f"Layer {i:>2} ({l.__class__.__name__:>10}): mean {mean:>12.7f} | std {std:.8f}| grad:data ratio {(std / p.std()).item():>13e}"  #  | grad:data ratio (mean) {(mean/p.mean()).item():>13e}"
    )
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].cpu().detach(), hy.cpu().detach())
    legends.append(f"layer {i}")
plt.legend(legends)

# %% visualize lr * grad.std/data.std for each parameter over time

plt.figure(figsize=(20, 4))
for i, p in enumerate(parameters):
    if p.ndim != 2:
        continue
    plt.plot([ud[j][i] for j in range(len(ud))])
plt.plot([0, len(ud)], [-3, -3], "k")

# %%
print(f"Train loss {forward(Xtr, Ytr, False):.4f}")
print(f"Train loss {forward(Xdev, Ydev, False):.4f}")

# %%


def predict_words():
    word = "." * block_size
    while True:
        x = torch.tensor([[char_to_int(x) for x in word[-3:]]])  # (1, 3)
        xenc = C[x].view(x.size(0), -1)  # (n, 3 * 27)
        ypred = xenc
        for p in layers:
            p.training = False
            ypred = p(ypred)
        ypred -= ypred.max()
        yprob = ypred.exp()
        yprob = yprob / yprob.sum(1, keepdim=True)
        word += CHARS[int(torch.multinomial(yprob, 1).item())]
        if word[-1] == ".":
            break
    return word[3:-1]


for i in range(30):
    print(predict_words())
