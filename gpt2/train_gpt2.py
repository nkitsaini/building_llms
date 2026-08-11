from dataclasses import dataclass

import plotly.graph_objects as go
import torch

import matplotlib.pyplot as plt
import typing as t
import torch.nn as nn
from torch.nn import functional as F
import plotly.express as px

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384


class CasualSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        # key, value, query
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

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

        fig = AutoFig()
        def plot(x, title: str = ""):
            # TODO: also try only seeing T-1 value.
            # fig = go.Figure()
            # fig.add_heatmap(y=x.squeeze().numpy(), name=title + "-heatmap")
            # go.Figure().add_histogram
            fig.add_histogram(y=x.contiguous().view(-1).numpy(), name=title + "-histogram")
            # fig.show()
        x = self.c_attn(x) # B, T, 3*C
        q, k, v = x.split(C, dim=2)
        # q = x[:, :, :C]
        # k = x[:, :, C:2*C]
        # v = x[:, :, 2*C:]
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)

        plot(k, 'k')
        plot(q, 'q')
        plot(v, 'v')
        # k.q for all previous T's
        # print(f"{k.shape=} {q.shape=}")
        # att = k @ q.transpose(-1, -2) * (k.size(-1)**-0.5) # [T, T]
        att = q @ k.transpose(-1, -2) * (k.size(-1)**-0.5) # [B, n_head, T, T]
        plot(q @ k.transpose(-1, -2), 'q@k')
        plot(att, 'q@k/sqrt(dk)')
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))  # ty: ignore[not-subscriptable]
        plot(att, 'qafter masked fill')
        # px.imshow(att[0, 0].numpy(), title="att[0][0] heatmap").show()
        fig.add_heatmap(z=att[0, 0].numpy(), name="att[0][0] heatmap")
        plot(att, 'qafter masked fill')
        att = F.softmax(att, dim=-1) # [B, n_head, T, T]

        all_close = torch.allclose(att.sum(-1), torch.ones_like(att.sum(-1)))
        assert all_close

        plot(att, 'after softmax')
        # print(f"{att.shape=} {v.shape=}")
        y = att @ v # (B, nH, T, T) @ (B, nH, T, C/nH) = (B, nH, T, C/nH)
        plot(y, 'after att@v')

        # without contiguous pytorch will throw error as `transpose` makes the tensor non-contiguous
        # because transpose does not reorder memory
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)
        plot(y, 'after c _proj')
        fig.show()
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
        fig = AutoFig()
        def plot(x, title: str = ""):
            # TODO: also try only seeing T-1 value.
            # fig = go.Figure()
            # fig.add_heatmap(y=x.squeeze().numpy(), name=title + "-heatmap")
            # go.Figure().add_histogram
            fig.add_histogram(y=x.view(-1).numpy(), name=title + "-histogram")
            # fig.show()
        # (B, T, n_embd)
        plot(x, 'input')
        xln1 = self.ln_1(x)
        plot(xln1, 'xln1')
        xattn = self.attn(xln1)
        plot(xattn, 'xattn')

        x = x + xattn
        # x = x + self.attn(self.ln_1(x))
        plot(x, 'after attn')
        xln2  = self.ln_2(x)
        plot(xln2, 'xln2')

        x = x + self.mlp(xln2)
        # x = x + self.mlp(self.ln_2(x))
        plot(x, 'after mlp')
        fig.show()
        return x # (B, T, n_embd?)


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor):
        # x = (B, T)
        # y = (B, T) for each T predict next token
        B, T = idx.shape

        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        fig = AutoFig()
        def plot(x, title: str = ""):
            # TODO: also try only seeing T-1 value.
            # fig = go.Figure()
            # fig.add_heatmap(y=x.squeeze().numpy(), name=title + "-heatmap")
            # px.imshow(x.squeeze().numpy(), title="before attention").show() # has character and range. Maybe fine.
            fig.add_histogram(y=x.view(-1).numpy(), name=title + "-histogram")

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # [0, ... T] shape=(T)
        pos_emb = self.transformer.wpe(pos)  # ty: ignore[call-non-callable] # (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # ty: ignore[call-non-callable] # (B, T, n_embd)

        plot(pos_emb, "pos_emb")
        plot(tok_emb, "tok_emb")

        # print(f"{pos_emb.shape=}, {tok_emb.shape=}") # [Checked]
        x = tok_emb + pos_emb # (B, T, n_emb) + (T, n_emb)
        # print(f"{x.shape=}") # [Checked]


        plot(x, "before attention")
        for i, block in enumerate(self.transformer.h):  # ty: ignore[invalid-argument-type]
            x = block(x) # (B, T, n_emb)?
            plot(x, f"after {i}th attention block")
            # px.imshow(x.squeeze().numpy(), title=f"after {i}th attention block").show() # ?
            """
            input: sigmoid-ish (mostly in range: -2, 2)
            out1: sigmoid-ish with outliers till (-50, 50)
            out2: sigmoid-ish with outliers till (~150)
            out3: sigmoid-ish with outliers till (~500)

            --- So somthing wrong in transformer
            """
            break

        fig.show()
        # px.imshow(x.squeeze().numpy()).show() # iffy, only a few embd places are active
        # print(f"{x.shape=}") # [Checked]
        x = self.transformer.ln_f(x)  # ty: ignore[call-non-callable]

        # print(f"{x.shape=}") # [Checked]
        # px.imshow(x.squeeze().numpy()).show() # iffy, only a few embd places are active
        logits = self.lm_head(x) # (B, T, embd) -> (B, T, vocab_size)
        # px.imshow(logits.squeeze().numpy()).show()
        # plt.imshow(logits.squeeze().numpy())
        # plt.show()

        # fig.imsho(logits.squeeze().numpy())
        # fig.show()

        # print(f"{x.shape=}") # [Checked]
        return logits

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

class AutoFig:
    def __init__(self):
        self.items = []
    def add_scatter(self, **kw):
        self.items.append(("scatter", kw))
    def add_heatmap(self, **kw):
        self.items.append(("heatmap", kw))
    def add_histogram(self, **kw):
        self.items.append(("histogram", kw))
    def show(self):
        from plotly.subplots import make_subplots
        titles = [kw.get('name', '-') for _, kw in self.items]
        fig = make_subplots(rows=len(self.items), cols=1, subplot_titles=titles)
        for i, (kind, kw) in enumerate(self.items, start=1):
            getattr(fig, f"add_{kind}")(row=i, col=1, **kw)
        fig.update_layout(height=300 * len(self.items))
        # fig.show()

def predict():
    num_return_sequences = 5
    max_length = 30

    model = GPT.from_pretrained('gpt2')
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
    fig = go.Figure()

    # with torch.no_grad():
    #     logits1 = model(x)
    #     logits2 = model(x)
    #     x2 = x.clone()
    #     x2[0][5] += 3
    #     logits3 = model(x2)
    #     lg1 = logits1[:, 4, :]
    #     lg2 = logits2[:, 4, :]
    #     lg3 = logits3[:, 4, :]
    #     fig = AutoFig()
    #     assert (lg1==lg2).all()
    #     assert (lg1==lg3).all()
    #     breakpoint()
    #     fig.show()
    #     return



    torch.manual_seed(42)
    while x.size(1) < max_length:
        with torch.no_grad():
            # print(f"{x.shape=}")
            logits = model(x)
            logits = logits[:, -1, :]
            # print(f"{logits.shape=}")
            # break
            # print(logits)
            # plt.plot(logits.squeeze().detach())
            fig.add_scatter(y=logits.squeeze().detach())
            probs = F.softmax(logits, dim=-1)

            topk_probs, topk_indices =torch.topk(probs, 50, dim=-1)

            ix = torch.multinomial(topk_probs, 1)
            xcol =torch.gather(topk_indices, -1, ix)
            x = torch.cat((x, xcol), dim=-1)

    # plt.show()
    # fig.show()
    # x = tokens.to('cuda')
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)

if __name__ == "__main__":
    predict()
    # model = GPT.from_pretrained('gpt2')
    # print("didn't crash!")
