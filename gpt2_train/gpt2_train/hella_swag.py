import os
import json
import requests
import tiktoken
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path
from transformers import GPT2LMHeadModel
from .dataset_helpers import get_hellaswag_dir


hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

output_dir = get_hellaswag_dir()
enc = tiktoken.get_encoding("gpt2")

def download_file(url: str, fpath: str | Path, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fpath, "wb") as file, tqdm(
        desc=str(fpath),
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

def file_path(split: str):
    return output_dir/f"hellaswag_{split}.jsonl"


def download(split: str):
    """Downloads HellaSwag DATA_CACHE_DIR"""
    data_url = hellaswags[split]
    data_filepath = file_path(split)
    if not os.path.exists(data_filepath):
        print(f"Downloading {data_url} to {data_filepath}...")
        download_file(data_url, data_filepath)
    else:
        print(f"File {data_filepath} already exists for split {split} from {data_url}")
def render_example(example):
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    data = {
        "ctx_tokens": None, # setup
        "ending_tokens": [], # options
        "label": label, # correct answer
    }

    # gather up all the tokens
    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = enc.encode(" " + end) # note: prepending " " because GPT-2 tokenizer
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    # Make the sizes same for final tokens
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return (
        data, # dict
        tokens, # tensor(4, max_len) => ctx + option + pad (pad=000)
        mask, # tensor(4, max_len) =>   000 + 111111 + 000
        label, # `int` correct answer index
    )

def iterate_examples(split):
    # there are 10,042 examples in total in val
    download(split)
    with open(file_path(split)) as f:
        for line in f:
            example = json.loads(line)
            yield example

@torch.no_grad()
def evaluate(model_type: str, device: str):

    torch.set_float32_matmul_precision('high') # use tf32
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)  # ty: ignore[invalid-argument-type]

    for example in iterate_examples("val"):
        data, tokens, mask, label = render_example(example)

        tokens = tokens.to(device)
        mask = mask.to(device)

        logits = model(tokens).logits # input: [B, T], output: [B, T, vocab_size]

        shift_logits = (logits[:, :-1, :]).contiguous() # [B, T-1, vocab_size]
        shift_tokens = (tokens[:, 1:]).contiguous() # [B, T-1]

        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none') # [B*(T-1)]
        shift_losses = shift_losses.view(tokens.size(0), -1)# [B, T-1]

        shift_mask = (mask[..., 1:])
