# %%

from IPython import get_ipython

_ip = get_ipython()
if _ip:
    _ip.run_line_magic("load_ext", "autoreload")
    _ip.run_line_magic("autoreload", "2")
# %%

from typing import overload
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F

from gpt.dataset import load_dataset

# from gpt import TinyShakespeareDataset


# %%

device = "cuda" if torch.cuda.is_available() else "cpu"
ds = load_dataset(device)
vocab_size = ds.vocab_size
#

print(ds.content[:100])
print("=========")
print(ds.chars)
print(ds.vocab_size)

ds.stoi
# %%
#
# ds.encode("hii there")

ds.tensor
len(ds.train_data), len(ds.val_data)

block_size = 8
n_embd = 32
batch_size = 32

torch.manual_seed(1337)
ds.get_batch("train", batch_size)
# ds.train_data[:block_size + 1]
# %%


class Head(nn.Module):
    def __init__(self, head_size: int = 16):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        B, T, C = input.shape
        q = self.query(input)
        k = self.key(input)
        # wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # ty:ignore[not-subscriptable]
        wei = torch.softmax(wei, dim=-1)
        v = self.value(input)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, total_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(total_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, input: torch.Tensor):
        x = torch.cat([h(input) for h in self.heads], dim=-1)
        x = self.proj(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
        )

    def forward(self, input: torch.Tensor):
        return self.net(input)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        head_size = n_embd // n_head
        self.head = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor):
        x = x + self.head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BiagramManualSeed(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # self.sa_head = Head(n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd, 4),
            Block(n_embd, 4),
            Block(n_embd, 4),
            nn.LayerNorm(n_embd),
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        _, T = idx.shape
        token_embeddings = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T, n_embd)
        x = pos_emb + token_embeddings  # B, T, head_size
        x = self.blocks(x)  # B, T, head_size
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int = 100) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -block_size:])  # B, T, C
            logits = logits[:, -1, :]  # B, C
            probs = F.softmax(logits, dim=-1)  # B, C
            idx_next = torch.multinomial(probs, num_samples=1)  # B, 1
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = BiagramManualSeed()
model = model.to(device)

model(*ds.get_batch("train", batch_size))


# %%
@torch.no_grad()
def estimate_loss(eval_iters: int = 20):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = ds.get_batch(split, batch_size)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# %%

idx = torch.zeros((1, 1)).long()
l = model.generate(idx, 100)[0]
print(ds.decode(l.tolist()))

# %%
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for i in tqdm(range(5000)):
    x, y = ds.get_batch("train", batch_size)
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print()
print(estimate_loss())


print(ds.decode(model.generate(torch.zeros((1, 1)).long(), 100)[0].tolist()))


# %%
"""
torch.manual_seed(1337)
B, T, C =4, 8, 32
x = torch.randn(B, T, C)


head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)


k = key(x) # (B, T, head_size)
q = query(x) # (B, T, head_size)
# (B, T, ...)
# (B, T, 1) # dot product with all past ones
#
wei = q @ k.transpose(-2, -1)


tril = torch.tril(torch.ones(T, T))
# wei = torch.zeros((T, T))
wei = wei.masked_fill(tril==0, float('-inf'))
wei = torch.softmax(wei, -1)

out = wei@ x
out.shape
"""
# %%
