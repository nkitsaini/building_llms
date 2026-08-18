"""

Launch:

uv run torchrun --standalone --nproc_per_node=8 -m gpt2_train.train_gpt2

"""
from pathlib import Path
from dataclasses import dataclass
import tiktoken

from gpt.dataset import load_dataset
from .plot_helpers import AutoFig
import plotly.graph_objects as go
import torch
import json

import os
import matplotlib.pyplot as plt
import math
import typing as t
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import plotly.express as px
import time
import inspect
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .dataset_helpers import get_fineweb_dir, get_hellaswag_dir, get_log_filepath, get_checkpoint_dir
from .hella_swag import render_example, iterate_examples


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

# ----------------- ddp init
ddp = int(os.environ.get("RANK", -1)) != -1

if ddp:
    assert torch.cuda.is_available(), "ddp without cuda?"  # AMD :(
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    device_type = 'cuda'
    master_process = ddp_rank == 0  # logging/checkpoint etc.
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    device = "cpu"
    device_type = 'cpu'
    if torch.cuda.is_available():
        device = "cuda"
        device_type = 'cuda'
# ----------------- ddp init end

def log(*args, **kwargs):
    if master_process:
        print(*args, **kwargs)

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

        x = self.c_attn(x)  # B, T, 3*C
        q, k, v = x.split(C, dim=2)
        # q = x[:, :, :C]
        # k = x[:, :, C:2*C]
        # v = x[:, :, 2*C:]
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

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
        self.c_proj.NANOGPT_SCALE_INIT = 1  # ty: ignore[invalid-assignment]

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
        xln2 = self.ln_2(x)
        x = x + self.mlp(xln2)
        return x  # (B, T, n_embd?)


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
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, target: torch.Tensor | None = None):
        # x = (B, T)
        # y = (B, T) for each T predict next token
        B, T = idx.shape

        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        )

        pos = torch.arange(
            0, T, dtype=torch.long, device=idx.device
        )  # [0, ... T] shape=(T)
        pos_emb = self.transformer.wpe(pos)  # ty: ignore[call-non-callable] # (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # ty: ignore[call-non-callable] # (B, T, n_embd)

        x = tok_emb + pos_emb  # (B, T, n_emb) + (T, n_emb)

        for i, block in enumerate(self.transformer.h):  # ty: ignore[invalid-argument-type]
            x = block(x)  # (B, T, n_emb)?

        x = self.transformer.ln_f(x)  # ty: ignore[call-non-callable]

        logits = self.lm_head(x)  # (B, T, embd) -> (B, T, vocab_size)
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(
        cls, model_type: t.Literal["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
    ) -> "GPT":
        from transformers import GPT2LMHeadModel

        print(f"loading weights from pretrained gpt: {model_type}")

        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M parameters
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M parameters
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M parameters
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M parameters
        }[model_type]
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)

        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard the predetermined attention "bias"

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_hf_keys = sd_hf.keys()
        sd_hf_keys = [
            k for k in sd_hf_keys if not k.endswith(".attn.bias")
        ]  # discard the predetermined attention "bias"
        sd_hf_keys = [
            k for k in sd_hf_keys if not k.endswith(".attn.masked_bias")
        ]  # ???

        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        assert len(sd_hf_keys) == len(sd_keys), (
            f"key length mismatch {len(sd_hf_keys)=} != {len(sd_keys)=}"
        )

        for k in sd_hf_keys:
            target = sd_hf[k]
            if any(k.endswith(w) for w in transposed):
                with torch.no_grad():
                    target = target.t()
            assert sd[k].shape == target.shape, f"{k}: {sd[k].shape=} != {target.shape}"
            with torch.no_grad():
                sd[k].copy_(target)
        return model

    def configure_optimizers(
        self, weight_decay: float, learning_rate: float, device: str | None = None
    ):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nondecay_params = [p for p in param_dict.values() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nondecay_params, "weight_decay": 0.0},
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nondecay_params = sum(p.numel() for p in nondecay_params)

        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nondecay_params)}, with {num_nondecay_params:,} parameters"
        )

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and (device is not None and "cuda" in device)
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        return optimizer

class FineWebDataLoaderLite:
    def __init__(self, B: int, T: int, process_rank: int, num_processes: int, split: str):
        assert split in ['train', 'val']
        self.B, self.T = B, T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split

        data_root = get_fineweb_dir()
        shards = list(data_root.iterdir())
        shards = [s for s in shards if split in s.name]
        self.shards = sorted(shards)
        assert len(shards) > 0, f"no shards found for split {split}"
        log(f"found {len(shards)} shards for split {split}")
        self.reset()


    def reset(self):
        self.current_shard = 0
        self.tokens = self.load_tokens(self.shards[self.current_shard])
        self.pos = self.B * self.T * self.process_rank

    def load_tokens(self, filepath: Path) -> torch.Tensor: # [T]
        npt = np.load(filepath)
        npt = npt.astype(np.int32) # ?? why? (karpathy added after video)
        return torch.tensor(npt, dtype=torch.long)

    def next_batch(self) -> t.Tuple[torch.Tensor, torch.Tensor]:
        B, T = self.B, self.T
        buf = self.tokens[self.pos : self.pos + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.pos += B * T * self.num_processes
        if self.pos + (B * T * self.num_processes + 1) > len(self.tokens):  # ???
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self.load_tokens(self.shards[self.current_shard])
            self.pos = self.B * self.T * self.process_rank

        return x, y


# Same as one in ./hella_swag.py
def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

class DataLoaderLite:
    def __init__(self, B: int, T: int, process_rank: int, num_processes: int):
        self.B, self.T = B, T
        self.process_rank = process_rank
        self.num_processes = num_processes

        self.data = load_dataset()
        enc = tiktoken.get_encoding("gpt2")
        self.tokens = torch.tensor(enc.encode(self.data.content))
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        self.pos = self.B * self.T * self.process_rank

    def next_batch(self) -> t.Tuple[torch.Tensor, torch.Tensor]:
        B, T = self.B, self.T
        buf = self.tokens[self.pos : self.pos + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.pos += B * T * self.num_processes
        if self.pos + (B * T * self.num_processes + 1) > len(self.tokens):  # ???
            self.pos = self.B * self.T * self.process_rank

        return x, y


def predict(model):
    num_return_sequences = 5
    max_length = 30

    model.eval()
    # model.to('cuda')
    #
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
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

            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
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
torch.set_float32_matmul_precision("high")


max_lr = 6e-4
min_lr = max_lr * 0.1

## for shakespear
# max_steps = 50
# warmup_steps = 10

## for fineweb (in accordance with gpt 2 training)
max_steps = 19073
warmup_steps = 715

def get_lr(it: int):

    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr

    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 1 -> 0
    return min_lr + coeff * (max_lr - min_lr)
    ...


def hella_swag_validate(*, model, step: int):
    num_correct_norm = 0
    num_total = 0
    for i, example in enumerate(iterate_examples("val")):
        # only process examples where i % ddp_world_size == ddp_rank
        if i % ddp_world_size != ddp_rank:
            continue
        # render the example into tokens and labels
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        # get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(tokens)
            pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct_norm += int(pred_norm == label)
    # reduce the stats across all processes
    if ddp:
        num_total = torch.tensor(num_total, dtype=torch.long, device=device)
        num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
        dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
        num_total = num_total.item()
        num_correct_norm = num_correct_norm.item()
    acc_norm = num_correct_norm / num_total
    if master_process:
        log(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
        log_file = get_log_filepath()
        with open(log_file, "a") as f:
            f.write(json.dumps({
                "step": step,
                "acc_norm": acc_norm
            }) + "\n")

def validate_and_checkpoint(*, val_loader: FineWebDataLoaderLite, model, raw_model, step: int):
    ## ========= Validation loss
    model.eval()
    val_loader.reset()
    with torch.no_grad():
        val_loss_accum = 0.0
        val_loss_steps = 20
        for _ in range(val_loss_steps):
            x, y = val_loader.next_batch()
            x = x.to(device)
            y = y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            val_loss_accum += (loss/val_loss_steps).detach()
    if ddp:
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
    if master_process:
        log(f"validation loss: {val_loss_accum.item():.4f}")
        log_file = get_log_filepath()
        with open(log_file, "a") as f:
            f.write(json.dumps({
                "val_loss": val_loss_accum.item(),
                "step": step,
            }) + "\n")
        ## ========= Checkpointing
        if step != 0 and (step % 5000 == 0 or step == max_steps -1):
            ck_path = get_checkpoint_dir()/f'model_{step:05d}.pt'
            torch.save( {
                "model": raw_model.state_dict(),
                'config': raw_model.config,
                'step': step,
                'val_loss': val_loss_accum.item()
            }, ck_path)

def main():
    global master_process, ddp


    if ddp:
        init_process_group(backend="nccl")  # What is nccl?
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)
    print(f"Using device: {device}")

    # total_batch_size = 524288 # (2^19)
    total_batch_size = 589824  # 12 * 1024 * (2^4 * 3)
    B = 8  # micro batch size (make as large as possible on device, doesn't affect quality)
    T = 1024  # sequence length

    assert total_batch_size % (B * T * ddp_world_size) == 0, (
        f"make sure {total_batch_size=} is divisble by {B*T*ddp_world_size=}"
    )
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

    if master_process:
        print(f"total desired batch_size: {total_batch_size:,}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    print("I am gpu", ddp_rank)

    # loader = DataLoaderLite(B, T, process_rank=ddp_rank, num_processes=ddp_world_size)
    train_loader = FineWebDataLoaderLite(B, T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
    val_loader = FineWebDataLoaderLite(B, T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")


    model = GPT(GPTConfig(vocab_size=50304))
    model.to(device)
    model = torch.compile(model)
    raw_model = model
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
        raw_model = model.module
    optimizer = raw_model.configure_optimizers(  # ty: ignore[unresolved-attribute]
        weight_decay=0.1, learning_rate=6e-4, device=device
    )
    # optimizer =torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)  # ty: ignore[unresolved-attribute]
    for step in range(max_steps):

        last_step = step == max_steps -1


        if step % 250 == 0 or last_step:
            validate_and_checkpoint(model=model, raw_model=raw_model, step=step, val_loader=val_loader)
            hella_swag_validate(model=model, step=step)
            predict(model)

        start = time.time()
        # reset_grad = step % grad_accum_steps == 0
        # optimize_grad = (step+1) % grad_accum_steps == 0

        # if reset_grad:
        model.train()  # ty: ignore[unresolved-attribute]
        optimizer.zero_grad()
        loss_accum = torch.zeros(1, device=device)
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x = x.to(device)
            y = y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                _, loss = model(x, y)
            # print(logits, loss)
            loss /= grad_accum_steps
            loss_accum += loss.detach()
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps -1)  # ty: ignore[unresolved-attribute]
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # ty: ignore[unresolved-attribute]
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # if optimize_grad:
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # wait for gpu operations to settle
        duration = time.time() - start
        tokens_processed = (train_loader.T * train_loader.B) * grad_accum_steps * ddp_world_size
        tps = (tokens_processed) / duration
        if master_process:
            log(
                f"step {step} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} norm: {norm:.4f} | duration: {duration:.3f}s | tps: {tps:.2f}"
            )

            with open(get_log_filepath(), "a") as f:
                f.write(json.dumps({
                    "step": step,
                    "loss_accum": loss_accum.item(),
                }) + "\n")
    if ddp:
        destroy_process_group()
    # logits, loss = model(x, y)
    # print(logits, loss)
    # print(x)
    # print(y)
    # predict()


if __name__ == "__main__":
    main()
    # model = GPT.from_pretrained('gpt2')
    # print("didn't crash!")
