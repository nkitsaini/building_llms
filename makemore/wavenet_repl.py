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
    vocab_size,
    END_TOKEN,
    START_TOKEN,
    char_to_int,
    create_count_tensor,
    create_training_data,
    read_words,
)
from makemore.nn import (
    cross_entropy,
    Linear,
    Model,
    Tanh,
    BatchNorm,
    Embedding,
    FlattenConsecutive,
    Sequential,
)

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
block_size = 8
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

sample_size = 5
for x, y in zip(Xtr[:sample_size], Ytr[:sample_size]):
    print("".join([CHARS[int(i.item())] for i in x]), "--> ", CHARS[y.item()])

# %%
embed_size = 32

g = torch.Generator(device).manual_seed(2147483649)

hidden_size = 68
# C = torch.randn(
#     len(CHARS), embed_size, generator=g
# )  # ; Linear(len(CHARS), embed_size, generator=g)
model = Sequential(
    [
        Embedding(vocab_size, embed_size, generator=g),
        FlattenConsecutive(2),
        Linear(2 * embed_size, hidden_size, bias=False),
        BatchNorm(hidden_size),
        Tanh(),
        # batch_size, vocab_size/2 = 4, hidden_size
        #
        FlattenConsecutive(2),
        Linear(2 * hidden_size, hidden_size, bias=False),
        BatchNorm(hidden_size),
        Tanh(),
        # batch_size, vocab_size/2 = 2, hidden_size
        FlattenConsecutive(2),
        Linear(2 * hidden_size, hidden_size, bias=False),
        BatchNorm(hidden_size),
        Tanh(),
        # batch_size, hidden_size
        Linear(hidden_size, vocab_size, bias=False),
    ]
)

with torch.no_grad():
    model.layers[-1].w *= 0.1  # ty:ignore[unresolved-attribute]

for p in model.parameters():
    p.requires_grad = True
print(sum(p.nelement() for p in model.parameters()))
# %%

# compiled_model = torch.compile(model)


def forward(x: torch.Tensor, y: torch.Tensor, training: bool = True):
    model.set_training(training)
    ypred = model(x)
    # ypred = compiled_model(x)
    loss = cross_entropy(ypred, y)
    return loss


def backprop(loss: torch.Tensor, lr: float = 1e-1):
    for l in model.layers:
        l.out.retain_grad()  # ty:ignore[unresolved-attribute]
    for p in model.parameters():
        p.grad = None
    loss.backward()
    for p in model.parameters():
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
        lr = 1e-1 if loop_num < 150000 else 0.01
        backprop(loss, lr)


# %%
losses = []
train(Xtr, Ytr, loops=200000)
# train(Xtr, Ytr, loops=1)
#
"""
See loss with batchnorm fix.
Previously ~1.93 on train and ~2.0 on test


"""
# %%

# %%
plt.plot(torch.tensor(losses).view(-1, 1000).sum(1))
losses[:2], losses[-2:]
# %%

with torch.no_grad():
    loss = forward(Xtr, Ytr, training=False)
    print("Train loss", loss)
    loss = forward(Xdev, Ydev, training=False)
    print("Validation loss", loss)

# %%


def predict_words():
    word = "." * block_size
    while True:
        x = torch.tensor([[char_to_int(x) for x in word[-3:]]])  # (1, 3)
        model.set_training(False)
        ypred = model(x)
        ypred -= ypred.max()
        yprob = ypred.exp()
        yprob = yprob / yprob.sum(1, keepdim=True)
        word += CHARS[int(torch.multinomial(yprob, 1).item())]
        if word[-1] == ".":
            break
    return word[3:-1]


for i in range(30):
    print(predict_words())
