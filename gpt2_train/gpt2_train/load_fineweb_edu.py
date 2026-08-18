# %%
import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm
import random
from pathlib import Path

local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"

shard_size = int(1e8) # 100M tokens per shard, total of 100 shards

# download the dataset
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")
# %%
if "__file__" in locals():
    output_dir = Path(__file__).parent.parent/'dataset'
else:
    # in jupyter notebook
    output_dir = Path(os.getcwd()) / 'gpt2_train'/'dataset'

# %%


idx = list(range(len(fw)))
random.shuffle(idx)
idx[:5]

# %%
enc = tiktoken.get_encoding('gpt2')
eot = enc._special_tokens["<|endoftext|>"]

def tokenize(doc: dict):
    text = doc['text']
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(text))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

# print (len(fw))
# print(fw[1])
# for i in fw:
#     print(i['text'])
#     break
