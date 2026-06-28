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
from makemore.biagram import CHARS, END_TOKEN, START_TOKEN, char_to_int, create_count_tensor, create_training_data, read_words
%matplotlib inline

import numpy as np

import torch

# %%
words = read_words()
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

hidden_size = 200
g = torch.Generator(device).manual_seed(2147483647)
C = torch.randn(27, embed_size, generator = g, device= device)
W1 = torch.randn(block_size * embed_size, hidden_size, generator = g,  device= device) * 0.2
B1 = torch.randn(hidden_size, generator = g,  device= device) * 0
W2 = torch.randn(hidden_size, 27, generator = g,  device= device) * 0.1
B2 = torch.randn(27, generator = g, device= device) * 0
bngain = torch.ones((1, hidden_size), device=device)
bnbias = torch.zeros((1, hidden_size), device=device)

bnmean_running = torch.zeros((1, hidden_size), device=device)
bnstd_running = torch.ones((1, hidden_size), device=device)

parameters = (C, W1, B1, W2, B2, bngain, bnbias)
sum(n.nelement() for n in parameters)
for p in parameters:
    p.requires_grad = True

# %%
track = {}
def calc_logits(xs: torch.Tensor, eval: bool = False) -> torch.Tensor:
    global bnmean_running
    global bnstd_running
    n = len(xs)
    # xs = (n, block_size)
    # C = (n_chars, embed_size)
    xenc = C[xs] # (n, block_size, embed_size)
    xenc = xenc.view(-1, block_size * embed_size) # (n, block_size * embed_size)
    track['xenc'] = xenc
    hpreact = (xenc @ W1) + B1
    bnmeani = hpreact.mean(0, keepdim=True)
    bnstdi = hpreact.std(0, keepdim=True)
    if eval:
        hpreact = (bngain *(hpreact - bnmean_running) / bnstd_running) + bnbias
    else:
        with torch.no_grad():
            bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
            bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi
        hpreact = (bngain *(hpreact - bnmeani) / bnstdi) + bnbias
    h1 = torch.tanh(hpreact) # (n, hidden_size)
    track['h1'] = h1
    return (h1 @ W2) + B2 # (n, 27)

def calc_loss(xs: torch.Tensor, ys: torch.Tensor, eval: bool = False):
    logits = calc_logits(xs, eval)
    track['logits'] = logits
    return F.cross_entropy(logits, ys)
    # logexp = logits.exp()
    # prob = logexp/logexp.sum(1, keepdim=True)
    # loss = - prob[torch.arange(n), ys].log().mean() # (n)
    # return loss
    #
def calc_loss_eval(xs: torch.Tensor, ys: torch.Tensor):
    with torch.no_grad():
        return calc_loss(xs, ys, True)

lossi = []

# %%

batch_size = 32

total_loops = 1
total_loops = 10_000
total_checkpoints = 10
lr = 0.1
lre = torch.linspace(-3, 0, total_loops)
lrs = 10**lre
lri = []

for i in range(total_loops):
    minibatch_idx = torch.randint(0, Xtr.size(0), (batch_size,), generator=g, device=device)
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
print(loss.item())
# plt.hist(track['h1'].view(-1).cpu().detach(), bins=20);
# plt.hist(track['logits'].view(-1).cpu().detach(), bins=20);
plt.figure(figsize=(16, 16))
plt.imshow(track['h1'][:30].cpu().detach() > 0.99, cmap='gray')

# %%
plt.subplot(121)
plt.hist(B1.grad.cpu().detach());
# plt.subplot(122)
# plt.hist(B2.grad.cpu().detach());
# %%

# F.cross_entropy(torch.zeros((1, 27)), torch.tensor([3]))

# %%
- np.log(1/27)

# %%
plt.plot(range(0, len(lossi)), lossi)
"""

"""

# %%
print(f"Training Loss {calc_loss_eval(Xtr, Ytr).item():.4f}")
print(f"Dev Loss {calc_loss_eval(Xdev, Ydev).item():.4f}")

# %%
"""
Run 1. loops=10k, lr=0.1

Training Loss 2.5776
Dev Loss 2.5890


Run 2. W2*0.1, B2*0

Training Loss 2.3274
Dev Loss 2.3553

Run3. W1*0.2, B1*0

Training Loss 2.2423
Dev Loss 2.2614


Run3. (Run2 + batchnorm)
Training Loss 2.2147
Dev Loss 2.2328

"""

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
        y_char = CHARS[y]
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
