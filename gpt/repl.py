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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ds = load_dataset(device)
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

torch.manual_seed(1337)
ds.get_batch('train')
# ds.train_data[:block_size + 1]
# %%

class BiagramManualSeed(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None=None):
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int = 100) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(idx) # B, T, C
            logits = logits[:, -1, :] # B, C
            probs = F.softmax(logits, dim=-1) # B, C
            idx_next = torch.multinomial(probs, num_samples=1) # B, 1
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

m = BiagramManualSeed(ds.vocab_size)
m = m.to(device)

m(*ds.get_batch('train'))

idx = torch.zeros((1, 1)).long()
l = m.generate(idx, 100)[0]
print(ds.decode(l.tolist()))

# %%
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32
for i in tqdm(range(10000)):
    x, y = ds.get_batch('train')
    logits, loss = m(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())


print(ds.decode(m.generate(torch.zeros((1, 1)).long(), 100)[0].tolist()))
