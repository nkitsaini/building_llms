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
xs, ys = create_training_data(words[:5], 3, debug = False)
xs.shape, xs.dtype, ys.shape, ys.dtype
# %%
C =torch.randn(27, 2)
# %%

xenc = F.one_hot(xs, num_classes=27).float()


# xenc[0]
# xenc[0] @ C

# C[0]

# %%
xs.shape
# %%
#
C[xs].flatten(1, 2).shape
# C[xs].unbind(1)

# %%
#
emb = C[xs].view(-1, 6)
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

mean_prob = (ys.float() @ prob).log().mean()
loss = - mean_prob

loss
