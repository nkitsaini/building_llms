from dataclasses import dataclass
import tiktoken

from gpt.dataset import load_dataset
from .plot_helpers import AutoFig
import plotly.graph_objects as go
import torch

import matplotlib.pyplot as plt
import math
import typing as t
import torch.nn as nn
from torch.nn import functional as F
import plotly.express as px
import time

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class CasualSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        # key, value, query
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # a lot of the times x += randn() happens inside residual stream
        #  if we scale by 1/sqrt(times) it remains gaussian
        self.c_proj.NANOGPT_SCALE_INIT = 1  # ty: ignore[invalid-assignment]

        self.n_head = config.n_head
        # not really bias, but matching gpt2 naming.
        # This is the matrix that make sure you can only
        # communicate with previous tokens
        # [T, T]
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )


    def forward(self, x: torch.Tensor):
        B, T, C = x.shape

        x = self.c_attn(x) # B, T, 3*C
        q, k, v = x.split(C, dim=2)
        # q = x[:, :, :C]
        # k = x[:, :, C:2*C]
        # v = x[:, :, 2*C:]
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)



        ### Non-flash attention
        # att = q @ k.transpose(-1, -2) * (k.size(-1)**-0.5) # [B, n_head, T, T]
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))  # ty: ignore[not-subscriptable]
        # att = F.softmax(att, dim=-1) # [B, n_head, T, T]

        # # all_close = torch.allclose(att.sum(-1), torch.ones_like(att.sum(-1)))
        # # assert all_close

        # y = att @ v # (B, nH, T, T) @ (B, nH, T, C/nH) = (B, nH, T, C/nH)
        #
        #
        ### Flas attentino
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # without contiguous pytorch will throw error as `transpose` makes the tensor non-contiguous
        # because transpose does not reorder memory
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        # key, value, query
        hidden = 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(hidden, config.n_embd)

    def forward(self, x: torch.Tensor):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)

        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor):
        xln1 = self.ln_1(x)
        xattn = self.attn(xln1)
        x = x + xattn
        xln2  = self.ln_2(x)
        x = x + self.mlp(xln2)
        return x # (B, T, n_embd?)


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        ...

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)


        # weight share scheme, but embedding (wte) is not matmul (lm_head)
        # so how does sharing work?
        self.transformer.wte.weight = self.lm_head.weight  # ty: ignore[invalid-assignment]
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2* self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, target: torch.Tensor | None = None):
        # x = (B, T)
        # y = (B, T) for each T predict next token
        B, T = idx.shape

        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # [0, ... T] shape=(T)
        pos_emb = self.transformer.wpe(pos)  # ty: ignore[call-non-callable] # (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # ty: ignore[call-non-callable] # (B, T, n_embd)

        x = tok_emb + pos_emb # (B, T, n_emb) + (T, n_emb)


        for i, block in enumerate(self.transformer.h):  # ty: ignore[invalid-argument-type]
            x = block(x) # (B, T, n_emb)?


        x = self.transformer.ln_f(x)  # ty: ignore[call-non-callable]

        logits = self.lm_head(x) # (B, T, embd) -> (B, T, vocab_size)
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type: t.Literal['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']) -> 'GPT':
        from transformers import GPT2LMHeadModel
        print(f"loading weights from pretrained gpt: {model_type}")


        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768), # 124M parameters
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024), # 350M parameters
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280), # 774M parameters
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600), # 1558M parameters
            }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024


        config = GPTConfig(**config_args)
        model = GPT(config)

        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [ k for k in sd_keys if not k.endswith('.attn.bias')] # discard the predetermined attention "bias"

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_hf_keys = sd_hf.keys()
        sd_hf_keys = [ k for k in sd_hf_keys if not k.endswith('.attn.bias')] # discard the predetermined attention "bias"
        sd_hf_keys = [ k for k in sd_hf_keys if not k.endswith('.attn.masked_bias')] # ???

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_hf_keys) == len(sd_keys), f"key length mismatch {len(sd_hf_keys)=} != {len(sd_keys)=}"

        for k in sd_hf_keys:
            target = sd_hf[k]
            if any(k.endswith(w) for w in transposed):
                with torch.no_grad():
                    target = target.t()
            assert sd[k].shape == target.shape, f"{k}: {sd[k].shape=} != {target.shape}"
            with torch.no_grad():
                sd[k].copy_(target)
        return model

class DataLoaderLite:
    def __init__(self, B, T):
        self.B, self.T = B, T

        self.data = load_dataset()
        enc = tiktoken.get_encoding('gpt2')
        self.tokens = torch.tensor(enc.encode(self.data.content))
        print(f'loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B*T)} batches')

        self.pos =0

    def next_batch(self) -> t.Tuple[torch.Tensor, torch.Tensor]:
        B, T = self.B, self.T
        buf = self.tokens[self.pos:self.pos+B*T +1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.pos += B*T
        if self.pos + B*T +1 > len(self.tokens):
            self.pos = 0

        return x, y

def predict():
    num_return_sequences = 5
    max_length = 30

    # model = GPT.from_pretrained('gpt2')
    model = GPT(GPTConfig())
    model.eval()
    # model.to('cuda')
    #
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    x = tokens

    import matplotlib.pyplot as plt

    torch.manual_seed(42)
    while x.size(1) < max_length:
        with torch.no_grad():
            logits = model(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            topk_probs, topk_indices =torch.topk(probs, 50, dim=-1)

            ix = torch.multinomial(topk_probs, 1)
            xcol =torch.gather(topk_indices, -1, ix)
            x = torch.cat((x, xcol), dim=-1)

    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)

"""

f32 - baseline
    step 21, loss: 6.412067 duration: 0.32 tps: 38352.09
tf32
    step 11, loss: 7.365175 duration: 0.26 tps: 47055.24
"""

# enable tf32
torch.set_float32_matmul_precision('high')


max_lr = 6e-4
min_lr = max_lr * 0.1
max_steps = 50
warmup_steps = 10
def get_lr(it: int):

    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr

    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # 1 -> 0
    return min_lr + coeff * (max_lr - min_lr)
    ...

def main():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)
    print(f"Using device: {device}")
    B, T = 12, 1024
    # B, T = 13, 1024
    loader = DataLoaderLite(B, T)
    model = GPT(GPTConfig(vocab_size=50304))
    model.to(device)
    model = torch.compile(model)
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4)
    # optimizer =torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)  # ty: ignore[unresolved-attribute]
    for step in range(max_steps):
        start = time.time()
        optimizer.zero_grad()
        x, y = loader.next_batch()
        x = x.to(device)
        y = y.to(device)
        if torch.cuda.is_available():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)
        # print(logits, loss)
        loss.backward()
        norm =torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # ty: ignore[unresolved-attribute]

        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize() # wait for gpu operations to settle
        duration = time.time() - start
        tps = (loader.T * loader.B)/duration
        print(f"step {step} | loss: {loss.item():.6f} | lr: {lr:.4e} norm: {norm:.4f} | duration: {duration:.3f}s | tps: {tps:.2f}")
        ...
    # logits, loss = model(x, y)
    # print(logits, loss)
    # print(x)
    # print(y)
    # predict()

if __name__ == "__main__":
    main()
    # model = GPT.from_pretrained('gpt2')
    # print("didn't crash!")
