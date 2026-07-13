# %% [markdown] id="rToK0Tku8PPn"
# ## makemore: becoming a backprop ninja

# %% id="8sFElPqq8PPp"
# there no change change in the first several cells from last lecture

# %% id="ChBbac4y8PPq"
from typing import cast
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt  # for making figures
from makemore.biagram import read_words

# %matplotlib inline

# %% id="x6GhEWW18aCS" colab={"base_uri": "https://localhost:8080/"} outputId="f5e23f10-d63e-4889-ee24-a315d5118b0f"
# download the names.txt file from github
# !wget https://raw.githubusercontent.com/karpathy/makemore/master/names.txt

# %% id="klmu3ZG08PPr" colab={"base_uri": "https://localhost:8080/"} outputId="ee52a759-d11e-48de-c959-1ebbf861320e"
# read in all the words
words = read_words()
print(len(words))
print(max(len(w) for w in words))
print(words[:8])

# %% id="BCQomLE_8PPs" colab={"base_uri": "https://localhost:8080/"} outputId="245ffcba-2553-4be6-b350-0631d1eff4d2"
# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)
print(itos)
print(vocab_size)

# %% id="V_zt2QHr8PPs" colab={"base_uri": "https://localhost:8080/"} outputId="e129de0c-3f30-4424-f60a-27add1aadad2"
# build the dataset
block_size = (
    3  # context length: how many characters do we take to predict the next one?
)


def build_dataset(words) -> tuple[torch.Tensor, torch.Tensor]:
    X, Y = [], []

    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]  # crop and append

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(X.shape, Y.shape)
    return X, Y


import random

random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])  # 80%
Xdev, Ydev = build_dataset(words[n1:n2])  # 10%
Xte, Yte = build_dataset(words[n2:])  # 10%


# %% id="eg20-vsg8PPt"
# ok biolerplate done, now we get to the action:


# %% id="MJPU8HT08PPu"
# utility function we will use later when comparing manual gradients to PyTorch gradients
def cmp(s, dt, t):
    assert dt.shape == t.shape, f"{dt.shape=} {t.shape=}"
    ex = torch.all(dt == t.grad).item()
    app = torch.allclose(dt, t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    print(
        f"{s:15s} | approximate: {str(app):5s} | exact: {str(ex):5s} | maxdiff: {maxdiff:4e}"
    )


# %% id="ZlFLjQyT8PPu" colab={"base_uri": "https://localhost:8080/"} outputId="e99ad7e8-fac3-410b-b737-f053c9f7d7da"
n_embd = 10  # the dimensionality of the character embedding vectors
n_hidden = 64  # the number of neurons in the hidden layer of the MLP

g = torch.Generator().manual_seed(2147483647)  # for reproducibility
C = torch.randn((vocab_size, n_embd), generator=g)
# Layer 1
W1 = cast(
    torch.Tensor,
    torch.randn((n_embd * block_size, n_hidden), generator=g)
    * (5 / 3)
    / ((n_embd * block_size) ** 0.5),
)
b1 = (
    torch.randn(n_hidden, generator=g) * 0.1
)  # using b1 just for fun, it's useless because of BN
# Layer 2
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.1
b2 = torch.randn(vocab_size, generator=g) * 0.1
# BatchNorm parameters
bngain = torch.randn((1, n_hidden)) * 0.1 + 1.0
bnbias = torch.randn((1, n_hidden)) * 0.1

# Note: I am initializating many of these parameters in non-standard ways
# because sometimes initializating with e.g. all zeros could mask an incorrect
# implementation of the backward pass.

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
print(sum(p.nelement() for p in parameters))  # number of parameters in total
for p in parameters:
    p.requires_grad = True

# %% id="QY-y96Y48PPv"
batch_size = 32
n = batch_size  # a shorter variable also, for convenience
# construct a minibatch
ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
Xb, Yb = Xtr[ix], Ytr[ix]  # batch X,Y

# %% id="8ofj1s6d8PPv" colab={"base_uri": "https://localhost:8080/"} outputId="04e2b0b5-6517-4351-fe6f-1a2230032ae6"
# forward pass, "chunkated" into smaller steps that are possible to backward one at a time

emb = C[Xb]  # embed the characters into vectors
embcat = emb.view(emb.shape[0], -1)  # concatenate the vectors
# Linear layer 1
hprebn = embcat @ W1 + b1  # hidden layer pre-activation
# BatchNorm layer
bnmeani = 1 / n * hprebn.sum(0, keepdim=True)
bndiff = hprebn - bnmeani
bndiff2 = bndiff**2
bnvar = (
    1 / (n - 1) * (bndiff2).sum(0, keepdim=True)
)  # note: Bessel's correction (dividing by n-1, not n)
bnvar_inv = (bnvar + 1e-5) ** -0.5
bnraw = bndiff * bnvar_inv
hpreact = bngain * bnraw + bnbias
# Non-linearity
h = torch.tanh(hpreact)  # hidden layer
# Linear layer 2
logits = h @ W2 + b2  # output layer
# cross entropy loss (same as F.cross_entropy(logits, Yb))
logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes  # subtract max for numerical stability
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdim=True)
counts_sum_inv = (
    counts_sum**-1
)  # if I use (1.0 / counts_sum) instead then I can't get backprop to be bit exact...
probs = counts * counts_sum_inv
logprobs = probs.log()
loss = -logprobs[range(n), Yb].mean()

# PyTorch backward pass
for p in parameters:
    p.grad = None
for t in [
    logprobs,
    probs,
    counts,
    counts_sum,
    counts_sum_inv,  # afaik there is no cleaner way
    norm_logits,
    logit_maxes,
    logits,
    h,
    hpreact,
    bnraw,
    bnvar_inv,
    bnvar,
    bndiff2,
    bndiff,
    hprebn,
    bnmeani,
    embcat,
    emb,
]:
    t.retain_grad()
loss.backward()
loss

# %% colab={"base_uri": "https://localhost:8080/"} id="NF0G72xL0pEm" outputId="6af55c2c-1b0b-42c3-bf74-64777c54d2f6"
logprobs.shape
counts.shape, counts_sum_inv.shape

# %% id="mO-8aqxK8PPw" colab={"base_uri": "https://localhost:8080/", "height": 106} outputId="e114011f-79a7-4705-8036-1156a5df714e"
# Exercise 1: backprop through the whole thing manually,
# backpropagating through exactly all of the variables
# as they are defined in the forward pass above, one by one

# -----------------
# YOUR CODE HERE :)
# -----------------

# loss (1) = -logprobs[range(n), Yb].mean() # logprobs.shape = (num_samples, vocab_size)

# just be (num_samples, vocab_size)
# 1. only the Yb's matter
# 2. everyone gets (1/sample_size weight)
dlogprobs = -F.one_hot(Yb, num_classes=27) / logprobs.size(0)
logprobs.shape, Yb.shape, logprobs[range(n), Yb].shape
# %%
dlogprobs
cmp("logprobs", dlogprobs, logprobs)

dprobs = dlogprobs * (1 / probs)
cmp("probs", dprobs, probs)

dcounts_sum_inv = (dprobs * counts).sum(1, keepdim=True)
cmp("counts_sum_inv", dcounts_sum_inv, counts_sum_inv)

dcounts_sum = dcounts_sum_inv * -(counts_sum ** (-2))
cmp("counts_sum", dcounts_sum, counts_sum)
#  probs = counts * counts_sum_inv
#  counts_sum = counts.sum(1, keepdims=True)
dcounts = dprobs * counts_sum_inv + dcounts_sum
cmp("counts", dcounts, counts)

dnorm_logits = dcounts * counts
cmp("norm_logits", dnorm_logits, norm_logits)

dlogit_maxes = -dnorm_logits.sum(
    1, keepdim=True
)  # subtract max for numerical stability
cmp("logit_maxes", dlogit_maxes, logit_maxes)

dlogits = dnorm_logits + dlogit_maxes * F.one_hot(
    logits.max(1).indices, num_classes=vocab_size
)
cmp("logits", dlogits, logits)

dh = dlogits @ W2.T
cmp("h", dh, h)

dW2 = h.T @ dlogits
cmp("W2", dW2, W2)

db2 = dlogits.sum(0)
cmp("b2", db2, b2)

dhpreact = dh * (1 - torch.square(h))
cmp("hpreact", dhpreact, hpreact)

dbngain = (dhpreact * bnraw).sum(0, keepdim=True)
cmp("bngain", dbngain, bngain)

dbnbias = dhpreact.sum(0, keepdim=True)
cmp("bnbias", dbnbias, bnbias)

dbnraw = dhpreact * bngain
cmp("bnraw", dbnraw, bnraw)

dbnvar_inv = (dbnraw * bndiff).sum(0, keepdim=True)
cmp("bnvar_inv", dbnvar_inv, bnvar_inv)

dbnvar = dbnvar_inv * (-0.5 * ((bnvar + 1e-5) ** -1.5))
cmp("bnvar", dbnvar, bnvar)


dbndiff2 = (dbnvar * (1 / (n - 1))).expand_as(bndiff2)
# bnvar = 1/(n-1)*(bndiff2).sum(0, keepdim=True) # note: Bessel's correction (dividing by n-1, not n)
cmp("bndiff2", dbndiff2, bndiff2)

dbndiff = dbnraw * bnvar_inv + dbndiff2 * 2 * bndiff
cmp("bndiff", dbndiff, bndiff)

dbnmeani = (dbndiff * (-1)).sum(0, keepdim=True)
# bndiff = hprebn - bnmeani
cmp("bnmeani", dbnmeani, bnmeani)

dhprebn = dbndiff + (dbnmeani * 1 / n).expand_as(hprebn)
cmp("hprebn", dhprebn, hprebn)
dembcat = dhprebn @ W1.T
cmp("embcat", dembcat, embcat)

dW1 = embcat.T @ dhprebn
cmp("W1", dW1, W1)

db1 = dhprebn.sum(0, keepdim=False)
cmp("b1", db1, b1)

demb = dembcat.view_as(emb)
cmp("emb", demb, emb)

# dC[Xb].shape, demb.shape, C.shape, Xb.shape
#
Xb_sized = Xb.view(-1)
demb_sized = demb.view(*Xb_sized.shape, -1)


dC = torch.zeros_like(C)
dC.index_add_(0, Xb_sized, demb_sized)
cmp("C", dC, C)
# %%

# %% id="ebLtYji_8PPw"
# Exercise 2: backprop through cross_entropy but all in one go
# to complete this challenge look at the mathematical expression of the loss,
# take the derivative, simplify the expression, and just write it out

# forward pass

# before:
# logit_maxes = logits.max(1, keepdim=True).values
# norm_logits = logits - logit_maxes # subtract max for numerical stability
# counts = norm_logits.exp()
# counts_sum = counts.sum(1, keepdims=True)
# counts_sum_inv = counts_sum**-1 # if I use (1.0 / counts_sum) instead then I can't get backprop to be bit exact...
# probs = counts * counts_sum_inv
# logprobs = probs.log()
# loss = -logprobs[range(n), Yb].mean()

# now:
loss_fast = F.cross_entropy(logits, Yb)
print(loss_fast.item(), "diff:", (loss_fast - loss).item())

# %% id="-gCXbB4C8PPx"
# backward pass

# -----------------
# YOUR CODE HERE :)
dlogits = None  # TODO. my solution is 3 lines
# -----------------

# cmp('logits', dlogits, logits) # I can only get approximate to be true, my maxdiff is 6e-9

# %% id="hd-MkhB68PPy"
# Exercise 3: backprop through batchnorm but all in one go
# to complete this challenge look at the mathematical expression of the output of batchnorm,
# take the derivative w.r.t. its input, simplify the expression, and just write it out
# BatchNorm paper: https://arxiv.org/abs/1502.03167

# forward pass

# before:
# bnmeani = 1/n*hprebn.sum(0, keepdim=True)
# bndiff = hprebn - bnmeani
# bndiff2 = bndiff**2
# bnvar = 1/(n-1)*(bndiff2).sum(0, keepdim=True) # note: Bessel's correction (dividing by n-1, not n)
# bnvar_inv = (bnvar + 1e-5)**-0.5
# bnraw = bndiff * bnvar_inv
# hpreact = bngain * bnraw + bnbias

# now:
hpreact_fast = (
    bngain
    * (hprebn - hprebn.mean(0, keepdim=True))
    / torch.sqrt(hprebn.var(0, keepdim=True, unbiased=True) + 1e-5)
    + bnbias
)
print("max diff:", (hpreact_fast - hpreact).abs().max())

# %% id="POdeZSKT8PPy"
# backward pass

# before we had:
# dbnraw = bngain * dhpreact
# dbndiff = bnvar_inv * dbnraw
# dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
# dbnvar = (-0.5*(bnvar + 1e-5)**-1.5) * dbnvar_inv
# dbndiff2 = (1.0/(n-1))*torch.ones_like(bndiff2) * dbnvar
# dbndiff += (2*bndiff) * dbndiff2
# dhprebn = dbndiff.clone()
# dbnmeani = (-dbndiff).sum(0)
# dhprebn += 1.0/n * (torch.ones_like(hprebn) * dbnmeani)

# calculate dhprebn given dhpreact (i.e. backprop through the batchnorm)
# (you'll also need to use some of the variables from the forward pass up above)

# -----------------
# YOUR CODE HERE :)
dhprebn = None  # TODO. my solution is 1 (long) line
# -----------------

cmp(
    "hprebn", dhprebn, hprebn
)  # I can only get approximate to be true, my maxdiff is 9e-10

# %% id="wPy8DhqB8PPz"
# Exercise 4: putting it all together!
# Train the MLP neural net with your own backward pass

# init
n_embd = 10  # the dimensionality of the character embedding vectors
n_hidden = 200  # the number of neurons in the hidden layer of the MLP

g = torch.Generator().manual_seed(2147483647)  # for reproducibility
C = torch.randn((vocab_size, n_embd), generator=g)
# Layer 1
W1 = (
    torch.randn((n_embd * block_size, n_hidden), generator=g)
    * (5 / 3)
    / ((n_embd * block_size) ** 0.5)
)
b1 = torch.randn(n_hidden, generator=g) * 0.1
# Layer 2
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.1
b2 = torch.randn(vocab_size, generator=g) * 0.1
# BatchNorm parameters
bngain = torch.randn((1, n_hidden)) * 0.1 + 1.0
bnbias = torch.randn((1, n_hidden)) * 0.1

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
print(sum(p.nelement() for p in parameters))  # number of parameters in total
for p in parameters:
    p.requires_grad = True

# same optimization as last time
max_steps = 200000
batch_size = 32
n = batch_size  # convenience
lossi = []

# use this context manager for efficiency once your backward pass is written (TODO)
# with torch.no_grad():

# kick off optimization
for i in range(max_steps):
    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]  # batch X,Y

    # forward pass
    emb = C[Xb]  # embed the characters into vectors
    embcat = emb.view(emb.shape[0], -1)  # concatenate the vectors
    # Linear layer
    hprebn = embcat @ W1 + b1  # hidden layer pre-activation
    # BatchNorm layer
    # -------------------------------------------------------------
    bnmean = hprebn.mean(0, keepdim=True)
    bnvar = hprebn.var(0, keepdim=True, unbiased=True)
    bnvar_inv = (bnvar + 1e-5) ** -0.5
    bnraw = (hprebn - bnmean) * bnvar_inv
    hpreact = bngain * bnraw + bnbias
    # -------------------------------------------------------------
    # Non-linearity
    h = torch.tanh(hpreact)  # hidden layer
    logits = h @ W2 + b2  # output layer
    loss = F.cross_entropy(logits, Yb)  # loss function

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()  # use this for correctness comparisons, delete it later!

    # manual backprop! #swole_doge_meme
    # -----------------
    # YOUR CODE HERE :)
    dC, dW1, db1, dW2, db2, dbngain, dbnbias = None, None, None, None, None, None, None
    grads = [dC, dW1, db1, dW2, db2, dbngain, dbnbias]
    # -----------------

    # update
    lr = 0.1 if i < 100000 else 0.01  # step learning rate decay
    for p, grad in zip(parameters, grads):
        p.data += (
            -lr * p.grad
        )  # old way of cheems doge (using PyTorch grad from .backward())
        # p.data += -lr * grad # new way of swole doge TODO: enable

    # track stats
    if i % 10000 == 0:  # print every once in a while
        print(f"{i:7d}/{max_steps:7d}: {loss.item():.4f}")
    lossi.append(loss.log10().item())

    if i >= 100:  # TODO: delete early breaking when you're ready to train the full net
        break

# %% id="ZEpI0hMW8PPz"
# useful for checking your gradients
# for p,g in zip(parameters, grads):
#   cmp(str(tuple(p.shape)), g, p)

# %% id="KImLWNoh8PP0"
# calibrate the batch norm at the end of training

with torch.no_grad():
    # pass the training set through
    emb = C[Xtr]
    embcat = emb.view(emb.shape[0], -1)
    hpreact = embcat @ W1 + b1
    # measure the mean/std over the entire training set
    bnmean = hpreact.mean(0, keepdim=True)
    bnvar = hpreact.var(0, keepdim=True, unbiased=True)


# %% id="6aFnP_Zc8PP0"
# evaluate train and val loss


@torch.no_grad()  # this decorator disables gradient tracking
def split_loss(split):
    x, y = {
        "train": (Xtr, Ytr),
        "val": (Xdev, Ydev),
        "test": (Xte, Yte),
    }[split]
    emb = C[x]  # (N, block_size, n_embd)
    embcat = emb.view(emb.shape[0], -1)  # concat into (N, block_size * n_embd)
    hpreact = embcat @ W1 + b1
    hpreact = bngain * (hpreact - bnmean) * (bnvar + 1e-5) ** -0.5 + bnbias
    h = torch.tanh(hpreact)  # (N, n_hidden)
    logits = h @ W2 + b2  # (N, vocab_size)
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())


split_loss("train")
split_loss("val")

# %% id="esWqmhyj8PP1"
# I achieved:
# train 2.0718822479248047
# val 2.1162495613098145

# %% id="xHeQNv3s8PP1"
# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    out = []
    context = [0] * block_size  # initialize with all ...
    while True:
        # forward pass
        emb = C[torch.tensor([context])]  # (1,block_size,d)
        embcat = emb.view(emb.shape[0], -1)  # concat into (N, block_size * n_embd)
        hpreact = embcat @ W1 + b1
        hpreact = bngain * (hpreact - bnmean) * (bnvar + 1e-5) ** -0.5 + bnbias
        h = torch.tanh(hpreact)  # (N, n_hidden)
        logits = h @ W2 + b2  # (N, vocab_size)
        # sample
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break

    print("".join(itos[i] for i in out))
